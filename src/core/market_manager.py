import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime
import torch
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class MarketManager:
    def __init__(self):
        self.exchanges = {}
        self.markets = {}
        self.active_pairs = set()
        self.orderbooks = {}
        self.market_data = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=32)  # For CPU-bound tasks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    async def initialize(self):
        """Initialize connections to exchanges and load market data"""
        # Initialize Kraken as primary exchange
        self.exchanges['kraken'] = ccxt.kraken({
            'enableRateLimit': True,
            'asyncio_loop': asyncio.get_event_loop()
        })
        
        # Add backup exchanges for arbitrage opportunities
        backup_exchanges = ['binance', 'coinbase', 'kucoin']
        for exchange_id in backup_exchanges:
            try:
                exchange_class = getattr(ccxt, exchange_id)
                self.exchanges[exchange_id] = exchange_class({
                    'enableRateLimit': True,
                    'asyncio_loop': asyncio.get_event_loop()
                })
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_id}: {e}")
        
        await self.load_markets()
        await self.start_market_watchers()

    async def load_markets(self):
        """Load and validate markets across all exchanges"""
        for exchange_id, exchange in self.exchanges.items():
            try:
                markets = await exchange.load_markets()
                self.markets[exchange_id] = markets
                
                # Find common trading pairs across exchanges for arbitrage
                if not self.active_pairs:
                    self.active_pairs = set(markets.keys())
                else:
                    self.active_pairs &= set(markets.keys())
                    
                logger.info(f"Loaded {len(markets)} markets from {exchange_id}")
            except Exception as e:
                logger.error(f"Error loading markets from {exchange_id}: {e}")

    async def start_market_watchers(self):
        """Start parallel market data collection for all active pairs"""
        tasks = []
        for pair in self.active_pairs:
            for exchange_id in self.exchanges:
                task = asyncio.create_task(self.watch_market(exchange_id, pair))
                tasks.append(task)
        
        await asyncio.gather(*tasks)

    async def watch_market(self, exchange_id: str, pair: str):
        """Watch a specific market pair on an exchange"""
        while True:
            try:
                orderbook = await self.exchanges[exchange_id].fetch_order_book(pair)
                await self.process_market_data(exchange_id, pair, orderbook)
                await asyncio.sleep(0.1)  # 100ms update interval
            except Exception as e:
                logger.error(f"Error watching {pair} on {exchange_id}: {e}")
                await asyncio.sleep(1)  # Back off on error

    async def process_market_data(self, exchange_id: str, pair: str, orderbook: dict):
        """Process market data using GPU acceleration when available"""
        try:
            # Convert orderbook to tensors for GPU processing
            bids = torch.tensor(orderbook['bids'], device=self.device)
            asks = torch.tensor(orderbook['asks'], device=self.device)
            
            # Calculate key metrics using GPU
            bid_vol = torch.sum(bids[:, 1])
            ask_vol = torch.sum(asks[:, 1])
            spread = asks[0][0] - bids[0][0]
            
            # Store processed data
            market_key = f"{exchange_id}:{pair}"
            self.orderbooks[market_key] = {
                'timestamp': datetime.now().timestamp(),
                'bids': bids.cpu().numpy(),
                'asks': asks.cpu().numpy(),
                'metrics': {
                    'bid_volume': bid_vol.item(),
                    'ask_volume': ask_vol.item(),
                    'spread': spread.item()
                }
            }
            
            # Check for arbitrage opportunities
            await self.check_arbitrage(pair)
            
        except Exception as e:
            logger.error(f"Error processing market data for {pair} on {exchange_id}: {e}")

    async def check_arbitrage(self, pair: str):
        """Check for arbitrage opportunities across exchanges"""
        prices = {}
        for exchange_id in self.exchanges:
            market_key = f"{exchange_id}:{pair}"
            if market_key in self.orderbooks:
                orderbook = self.orderbooks[market_key]
                prices[exchange_id] = {
                    'bid': orderbook['bids'][0][0],
                    'ask': orderbook['asks'][0][0]
                }
        
        # Find potential arbitrage opportunities
        for ex1 in prices:
            for ex2 in prices:
                if ex1 != ex2:
                    profit = prices[ex1]['bid'] - prices[ex2]['ask']
                    if profit > 0:
                        logger.info(f"Arbitrage opportunity: {pair} "
                                  f"Buy on {ex2} at {prices[ex2]['ask']}, "
                                  f"Sell on {ex1} at {prices[ex1]['bid']}, "
                                  f"Potential profit: {profit}")
                        # Signal trading opportunity to execution system
                        await self.signal_arbitrage(pair, ex1, ex2, profit)

    async def signal_arbitrage(self, pair: str, sell_ex: str, buy_ex: str, profit: float):
        """Signal an arbitrage opportunity to the trading system"""
        # Implementation will be connected to the trade executor
        pass

    async def get_all_markets(self) -> Dict:
        """Get current status of all active markets"""
        return {
            'active_pairs': list(self.active_pairs),
            'orderbooks': self.orderbooks,
            'exchanges': list(self.exchanges.keys())
        }

    async def get_market_depth(self, pair: str, exchanges: Optional[List[str]] = None) -> Dict:
        """Get market depth data for a specific pair across exchanges"""
        depth_data = {}
        target_exchanges = exchanges if exchanges else self.exchanges.keys()
        
        for exchange_id in target_exchanges:
            market_key = f"{exchange_id}:{pair}"
            if market_key in self.orderbooks:
                depth_data[exchange_id] = self.orderbooks[market_key]
        
        return depth_data

    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown()
        for exchange in self.exchanges.values():
            asyncio.create_task(exchange.close())

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import torch
import numpy as np

from core.config import load_config
from core.market_manager import MarketManager
from core.trade_executor import TradeExecutor
from core.risk_manager import RiskManager
from models.deep_learning import DeepLearningModel
from models.market_predictor import MarketPredictor
from strategies.multi_market import MultiMarketStrategy
from data.market_data import MarketDataManager

logger = logging.getLogger(__name__)

class SystemController:
    def __init__(self):
        self.config = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.market_manager = MarketManager()
        self.trade_executor = TradeExecutor()
        self.risk_manager = RiskManager()
        self.market_data = MarketDataManager()
        self.deep_learning = DeepLearningModel(device=self.device)
        self.market_predictor = MarketPredictor()
        self.strategy = MultiMarketStrategy()
        
        # System state
        self.is_running = False
        self.active_models = {}
        self.market_states = {}
        self.trading_signals = {}
        
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing trading system...")
            
            # Initialize market data system
            await self.market_data.initialize()
            
            # Initialize market manager
            await self.market_manager.initialize()
            
            # Load trading pairs
            self.trading_pairs = self.config['trading']['trading_pairs']
            
            # Initialize models for each trading pair
            for exchange in self.trading_pairs:
                for pair in self.trading_pairs[exchange]:
                    await self.deep_learning.initialize_model(pair)
                    
            logger.info("Trading system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            raise

    async def start(self):
        """Start the trading system"""
        try:
            self.is_running = True
            
            # Start main processing loops
            await asyncio.gather(
                self._market_data_loop(),
                self._analysis_loop(),
                self._trading_loop(),
                self._model_update_loop()
            )
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            self.is_running = False
            raise

    async def _market_data_loop(self):
        """Process market data updates"""
        while self.is_running:
            try:
                for exchange in self.trading_pairs:
                    for pair in self.trading_pairs[exchange]:
                        # Fetch and process market data
                        market_data = await self.market_manager.get_market_depth(pair)
                        if market_data:
                            await self.market_data.process_market_data(pair, market_data)
                
                await asyncio.sleep(0.01)  # 10ms interval
                
            except Exception as e:
                logger.error(f"Error in market data loop: {e}")
                await asyncio.sleep(1)

    async def _analysis_loop(self):
        """Analyze market conditions and generate predictions"""
        while self.is_running:
            try:
                # Get latest market states
                market_states = await self.market_data.get_updates()
                
                # Generate predictions for each market
                predictions = {}
                for symbol in market_states:
                    pred = await self.market_predictor.predict(
                        symbol,
                        market_states[symbol]
                    )
                    predictions[symbol] = pred
                
                # Analyze trading opportunities
                opportunities = await self.strategy.analyze_opportunities(market_states)
                
                # Update trading signals
                self.trading_signals = {
                    opp['symbol']: opp for opp in opportunities
                }
                
                await asyncio.sleep(0.1)  # 100ms interval
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(1)

    async def _trading_loop(self):
        """Execute trading decisions"""
        while self.is_running:
            try:
                # Process trading signals
                for symbol, signal in self.trading_signals.items():
                    # Validate trade with risk manager
                    if await self.risk_manager.validate_trade(signal):
                        # Execute trade
                        await self.trade_executor.execute_trade(signal)
                
                await asyncio.sleep(0.05)  # 50ms interval
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(1)

    async def _model_update_loop(self):
        """Update machine learning models"""
        while self.is_running:
            try:
                current_time = datetime.now()
                update_interval = self.config['ml']['model_update_interval']
                
                for exchange in self.trading_pairs:
                    for pair in self.trading_pairs[exchange]:
                        # Get training data
                        market_data = await self.market_data.get_historical_data(
                            pair,
                            start_time=current_time - timedelta(days=30)
                        )
                        
                        # Update models
                        await self.market_predictor.train(pair, market_data)
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in model update loop: {e}")
                await asyncio.sleep(60)

    async def get_system_status(self) -> Dict:
        """Get current system status"""
        try:
            # Collect status from all components
            status = {
                'running': self.is_running,
                'gpu_device': str(self.device),
                'active_pairs': len(self.trading_pairs),
                'active_models': len(self.active_models),
                'market_states': len(self.market_states),
                'trading_signals': len(self.trading_signals),
                'last_update': datetime.now().isoformat()
            }
            
            # Get portfolio status
            portfolio = await self.trade_executor.get_portfolio_status()
            status['portfolio'] = portfolio
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

    async def stop(self):
        """Stop the trading system"""
        try:
            logger.info("Stopping trading system...")
            self.is_running = False
            
            # Cleanup resources
            self.market_manager.cleanup()
            self.market_data.cleanup()
            self.deep_learning.cleanup()
            self.market_predictor.cleanup()
            self.strategy.cleanup()
            
            logger.info("Trading system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
            raise

    async def handle_error(self, error: Exception):
        """Handle system errors"""
        try:
            logger.error(f"System error: {error}")
            
            # Implement error recovery logic
            if isinstance(error, RuntimeError):
                # Hardware-related errors
                if "CUDA" in str(error):
                    logger.info("Attempting GPU recovery...")
                    torch.cuda.empty_cache()
                    await self.initialize()
            elif isinstance(error, ConnectionError):
                # Network-related errors
                logger.info("Attempting network recovery...")
                await asyncio.sleep(5)
                await self.initialize()
            else:
                # Unknown errors
                logger.error("Unknown error occurred. Stopping system...")
                await self.stop()
            
        except Exception as e:
            logger.error(f"Error in error handler: {e}")
            await self.stop()
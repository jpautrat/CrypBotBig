import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import logging
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import redis
from concurrent.futures import ThreadPoolExecutor
from collections import deque

logger = logging.getLogger(__name__)

class MarketDataManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thread_pool = ThreadPoolExecutor(max_workers=32)
        self.data_cache = {}
        self.market_states = {}
        self.feature_buffers = {}
        self.volatility_windows = {}
        
        # Initialize InfluxDB client for time-series data
        self.influx_client = InfluxDBClient(
            url="http://localhost:8086",
            token="your-token",
            org="trading-system"
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
        # Initialize Redis for real-time data caching
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        # Initialize feature calculation on GPU
        self.setup_gpu_tensors()

    def setup_gpu_tensors(self):
        """Initialize GPU tensors for fast feature calculation"""
        # Pre-allocate tensors for common calculations
        self.ma_weights = {}
        self.volatility_masks = {}
        
        # Common moving average windows
        windows = [5, 10, 20, 50, 100, 200]
        
        for window in windows:
            # Create exponential weights for EMA calculation
            alpha = 2.0 / (window + 1)
            weights = torch.tensor(
                [(1 - alpha) ** i for i in range(window)],
                device=self.device
            )
            self.ma_weights[window] = weights / weights.sum()
            
            # Create masks for rolling volatility
            self.volatility_masks[window] = torch.ones(window, device=self.device)

    async def initialize(self):
        """Initialize database connections and data structures"""
        try:
            # Create InfluxDB bucket if it doesn't exist
            buckets_api = self.influx_client.buckets_api()
            if "market_data" not in [b.name for b in buckets_api.find_buckets()]:
                buckets_api.create_bucket(bucket_name="market_data", org="trading-system")
            
            logger.info("Market data manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing market data manager: {e}")
            raise

    async def process_market_data(self, symbol: str, data: Dict):
        """Process incoming market data using GPU acceleration"""
        try:
            # Convert data to tensors
            price_tensor = torch.tensor(data['price'], device=self.device)
            volume_tensor = torch.tensor(data['volume'], device=self.device)
            
            # Update feature buffers
            if symbol not in self.feature_buffers:
                self.feature_buffers[symbol] = {
                    'price': deque(maxlen=200),
                    'volume': deque(maxlen=200)
                }
            
            self.feature_buffers[symbol]['price'].append(price_tensor)
            self.feature_buffers[symbol]['volume'].append(volume_tensor)
            
            # Calculate features if we have enough data
            if len(self.feature_buffers[symbol]['price']) >= 20:
                features = await self.calculate_features(symbol)
                
                # Store processed data
                await self.store_processed_data(symbol, features)
                
                # Update market state
                self.market_states[symbol] = {
                    'timestamp': datetime.now().timestamp(),
                    'features': features
                }
                
                # Cache recent data in Redis
                self.cache_market_data(symbol, features)
                
        except Exception as e:
            logger.error(f"Error processing market data for {symbol}: {e}")

    async def calculate_features(self, symbol: str) -> Dict:
        """Calculate technical indicators using GPU acceleration"""
        try:
            # Convert buffer to tensor
            prices = torch.tensor(
                list(self.feature_buffers[symbol]['price']),
                device=self.device
            )
            volumes = torch.tensor(
                list(self.feature_buffers[symbol]['volume']),
                device=self.device
            )
            
            # Calculate moving averages
            features = {}
            for window in self.ma_weights:
                if len(prices) >= window:
                    # Calculate EMA using convolution
                    ema = torch.conv1d(
                        prices.view(1, 1, -1),
                        self.ma_weights[window].view(1, 1, -1),
                        padding=window-1
                    )
                    features[f'ema_{window}'] = ema[0, 0, -1].item()
            
            # Calculate volatility
            returns = torch.log(prices[1:] / prices[:-1])
            volatility = torch.std(returns).item()
            
            # Calculate volume profile
            vol_ema = torch.conv1d(
                volumes.view(1, 1, -1),
                self.ma_weights[20].view(1, 1, -1),
                padding=19
            )
            
            # Calculate momentum indicators
            roc = (prices[-1] / prices[0] - 1).item()
            
            # Store calculated features
            features.update({
                'volatility': volatility,
                'volume_ema': vol_ema[0, 0, -1].item(),
                'roc': roc,
                'current_price': prices[-1].item(),
                'timestamp': datetime.now().timestamp()
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol}: {e}")
            return {}

    async def store_processed_data(self, symbol: str, features: Dict):
        """Store processed market data in InfluxDB"""
        try:
            point = Point("market_data")\
                .tag("symbol", symbol)\
                .time(datetime.utcnow())
            
            for feature_name, value in features.items():
                point = point.field(feature_name, value)
            
            self.write_api.write(bucket="market_data", record=point)
            
        except Exception as e:
            logger.error(f"Error storing processed data for {symbol}: {e}")

    def cache_market_data(self, symbol: str, features: Dict):
        """Cache recent market data in Redis"""
        try:
            # Store features in Redis with 1-hour expiration
            self.redis_client.hmset(
                f"market_data:{symbol}",
                {k: str(v) for k, v in features.items()}
            )
            self.redis_client.expire(f"market_data:{symbol}", 3600)
            
        except Exception as e:
            logger.error(f"Error caching market data for {symbol}: {e}")

    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime = None
    ) -> pd.DataFrame:
        """Retrieve historical data from InfluxDB"""
        try:
            query = f'''
                from(bucket: "market_data")
                    |> range(start: {start_time.isoformat()}Z
                    {f', stop: {end_time.isoformat()}Z' if end_time else ''})
                    |> filter(fn: (r) => r["symbol"] == "{symbol}")
            '''
            
            result = self.influx_client.query_api().query_data_frame(query=query)
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def get_market_state(self, symbol: str) -> Dict:
        """Get current market state including technical indicators"""
        try:
            # Try to get from Redis cache first
            cached_data = self.redis_client.hgetall(f"market_data:{symbol}")
            
            if cached_data:
                return {
                    k.decode(): float(v.decode())
                    for k, v in cached_data.items()
                }
            
            # Fall back to latest state in memory
            return self.market_states.get(symbol, {})
            
        except Exception as e:
            logger.error(f"Error getting market state for {symbol}: {e}")
            return {}

    async def get_updates(self) -> Dict:
        """Get all recent market updates"""
        return {
            symbol: state
            for symbol, state in self.market_states.items()
            if datetime.now().timestamp() - state['timestamp'] < 60
        }

    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown()
        self.influx_client.close()
        self.redis_client.close()
        torch.cuda.empty_cache()
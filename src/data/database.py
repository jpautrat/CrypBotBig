import asyncpg
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config: Dict):
        self.config = config
        self.pool = None
        self.influx_client = None
        self.write_api = None
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Initialize PostgreSQL connection pool
            self.pool = await asyncpg.create_pool(
                self.config['system']['database_url'],
                min_size=5,
                max_size=20
            )
            
            # Initialize InfluxDB client
            self.influx_client = InfluxDBClient(
                url=self.config['system']['influxdb_url'],
                token=self.config['system'].get('influxdb_token'),
                org="trading-system"
            )
            self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
            
            # Create tables if they don't exist
            await self._initialize_tables()
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database connections: {e}")
            raise
            
    async def _initialize_tables(self):
        """Initialize database tables"""
        try:
            # Read and execute schema.sql
            with open('src/data/schema.sql', 'r') as f:
                schema = f.read()
                
            async with self.pool.acquire() as conn:
                await conn.execute(schema)
                
            logger.info("Database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
            raise
            
    async def store_market_data(self, data: Dict):
        """Store market data in both PostgreSQL and InfluxDB"""
        try:
            # Store in PostgreSQL for historical analysis
            query = """
                INSERT INTO market_data (
                    symbol, exchange, timestamp, price, volume,
                    open, high, low, close
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    data['symbol'],
                    data['exchange'],
                    data['timestamp'],
                    data['price'],
                    data['volume'],
                    data['open'],
                    data['high'],
                    data['low'],
                    data['close']
                )
            
            # Store in InfluxDB for real-time analysis
            point = Point("market_data")\
                .tag("symbol", data['symbol'])\
                .tag("exchange", data['exchange'])\
                .field("price", float(data['price']))\
                .field("volume", float(data['volume']))\
                .time(data['timestamp'])
                
            self.write_api.write(bucket="market_data", record=point)
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            
    async def store_trade(self, trade: Dict):
        """Store trade execution details"""
        try:
            query = """
                INSERT INTO trades (
                    symbol, exchange, side, price, amount, cost,
                    fee, timestamp, order_id, strategy,
                    prediction_confidence, market_score
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
            """
            
            async with self.pool.acquire() as conn:
                trade_id = await conn.fetchval(
                    query,
                    trade['symbol'],
                    trade['exchange'],
                    trade['side'],
                    trade['price'],
                    trade['amount'],
                    trade['cost'],
                    trade['fee'],
                    trade['timestamp'],
                    trade.get('order_id'),
                    trade.get('strategy'),
                    trade.get('prediction_confidence'),
                    trade.get('market_score')
                )
                
            return trade_id
            
        except Exception as e:
            logger.error(f"Error storing trade: {e}")
            
    async def store_portfolio_snapshot(self, snapshot: Dict):
        """Store portfolio snapshot"""
        try:
            query = """
                INSERT INTO portfolio_snapshots (
                    timestamp, total_value_usd, realized_pnl,
                    unrealized_pnl, risk_exposure
                ) VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """
            
            async with self.pool.acquire() as conn:
                snapshot_id = await conn.fetchval(
                    query,
                    snapshot['timestamp'],
                    snapshot['total_value_usd'],
                    snapshot['realized_pnl'],
                    snapshot['unrealized_pnl'],
                    snapshot['risk_exposure']
                )
                
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Error storing portfolio snapshot: {e}")
            
    async def store_model_prediction(self, prediction: Dict):
        """Store model predictions"""
        try:
            query = """
                INSERT INTO model_predictions (
                    symbol, timestamp, horizon_minutes,
                    predicted_price, confidence_score
                ) VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """
            
            async with self.pool.acquire() as conn:
                prediction_id = await conn.fetchval(
                    query,
                    prediction['symbol'],
                    prediction['timestamp'],
                    prediction['horizon_minutes'],
                    prediction['predicted_price'],
                    prediction['confidence_score']
                )
                
            return prediction_id
            
        except Exception as e:
            logger.error(f"Error storing model prediction: {e}")
            
    async def update_prediction_accuracy(self, prediction_id: int, actual_price: float):
        """Update prediction with actual price and error margin"""
        try:
            query = """
                UPDATE model_predictions
                SET actual_price = $1,
                    error_margin = ABS($1 - predicted_price) / predicted_price
                WHERE id = $2
            """
            
            async with self.pool.acquire() as conn:
                await conn.execute(query, actual_price, prediction_id)
                
        except Exception as e:
            logger.error(f"Error updating prediction accuracy: {e}")
            
    async def store_market_analysis(self, analysis: Dict):
        """Store market analysis results"""
        try:
            query = """
                INSERT INTO market_analysis (
                    symbol, timestamp, volatility, trend_strength,
                    market_regime, trading_signal, correlation_score,
                    volume_profile
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
            """
            
            async with self.pool.acquire() as conn:
                analysis_id = await conn.fetchval(
                    query,
                    analysis['symbol'],
                    analysis['timestamp'],
                    analysis['volatility'],
                    analysis['trend_strength'],
                    analysis['market_regime'],
                    analysis.get('trading_signal'),
                    analysis.get('correlation_score'),
                    analysis.get('volume_profile')
                )
                
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error storing market analysis: {e}")
            
    async def store_system_metrics(self, metrics: Dict):
        """Store system performance metrics"""
        try:
            query = """
                INSERT INTO system_metrics (
                    timestamp, cpu_usage, memory_usage,
                    gpu_usage, gpu_memory_usage, api_latency_ms,
                    active_models, prediction_time_ms
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    metrics['timestamp'],
                    metrics['cpu_usage'],
                    metrics['memory_usage'],
                    metrics['gpu_usage'],
                    metrics['gpu_memory_usage'],
                    metrics['api_latency_ms'],
                    metrics['active_models'],
                    metrics['prediction_time_ms']
                )
                
        except Exception as e:
            logger.error(f"Error storing system metrics: {e}")
            
    async def log_error(self, error: Dict):
        """Log system errors"""
        try:
            query = """
                INSERT INTO error_logs (
                    timestamp, component, error_type,
                    error_message, stack_trace
                ) VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """
            
            async with self.pool.acquire() as conn:
                error_id = await conn.fetchval(
                    query,
                    error['timestamp'],
                    error['component'],
                    error['error_type'],
                    error['error_message'],
                    error.get('stack_trace')
                )
                
            return error_id
            
        except Exception as e:
            logger.error(f"Error logging error: {e}")
            
    async def get_market_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime = None
    ) -> pd.DataFrame:
        """Get historical market data"""
        try:
            query = """
                SELECT *
                FROM market_data
                WHERE symbol = $1
                    AND timestamp >= $2
                    {end_condition}
                ORDER BY timestamp ASC
            """
            
            end_condition = "AND timestamp <= $3" if end_time else ""
            query = query.format(end_condition=end_condition)
            
            params = [symbol, start_time]
            if end_time:
                params.append(end_time)
                
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            return pd.DataFrame(rows, columns=[
                'id', 'symbol', 'exchange', 'timestamp', 'price',
                'volume', 'open', 'high', 'low', 'close', 'created_at'
            ])
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
            
    async def get_portfolio_performance(
        self,
        start_time: datetime,
        end_time: datetime = None
    ) -> Dict:
        """Get portfolio performance metrics"""
        try:
            query = """
                SELECT *
                FROM portfolio_snapshots
                WHERE timestamp >= $1
                    {end_condition}
                ORDER BY timestamp ASC
            """
            
            end_condition = "AND timestamp <= $2" if end_time else ""
            query = query.format(end_condition=end_condition)
            
            params = [start_time]
            if end_time:
                params.append(end_time)
                
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            snapshots = pd.DataFrame(rows)
            
            # Calculate performance metrics
            if not snapshots.empty:
                initial_value = snapshots['total_value_usd'].iloc[0]
                final_value = snapshots['total_value_usd'].iloc[-1]
                returns = snapshots['total_value_usd'].pct_change()
                
                performance = {
                    'total_return': (final_value - initial_value) / initial_value,
                    'sharpe_ratio': np.mean(returns) / np.std(returns) if len(returns) > 1 else 0,
                    'max_drawdown': self._calculate_max_drawdown(snapshots['total_value_usd']),
                    'profit_factor': self._calculate_profit_factor(snapshots)
                }
                
                return performance
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting portfolio performance: {e}")
            return {}
            
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown from a series of values"""
        peak = values.expanding(min_periods=1).max()
        drawdown = (values - peak) / peak
        return abs(drawdown.min())
        
    def _calculate_profit_factor(self, snapshots: pd.DataFrame) -> float:
        """Calculate profit factor from portfolio snapshots"""
        pnl = snapshots['realized_pnl']
        profits = pnl[pnl > 0].sum()
        losses = abs(pnl[pnl < 0].sum())
        return profits / losses if losses != 0 else float('inf')
        
    async def cleanup(self):
        """Cleanup database connections"""
        try:
            await self.pool.close()
            self.influx_client.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up database connections: {e}")

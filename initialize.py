import asyncio
import asyncpg
import logging
from influxdb_client import InfluxDBClient
import redis
import os
from core.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def initialize_postgres():
    """Initialize PostgreSQL database"""
    try:
        config = load_config()
        conn = await asyncpg.connect(config['system']['database_url'])
        
        # Read schema file
        with open('src/data/schema.sql', 'r') as f:
            schema = f.read()
            
        # Execute schema
        await conn.execute(schema)
        logger.info("PostgreSQL database initialized successfully")
        
        await conn.close()
        
    except Exception as e:
        logger.error(f"Error initializing PostgreSQL: {e}")
        raise

async def initialize_influxdb():
    """Initialize InfluxDB"""
    try:
        config = load_config()
        client = InfluxDBClient(
            url=config['system']['influxdb_url'],
            token=config['system']['influxdb_token'],
            org="trading-system"
        )
        
        # Create bucket if it doesn't exist
        buckets_api = client.buckets_api()
        if "market_data" not in [b.name for b in buckets_api.find_buckets()]:
            buckets_api.create_bucket(
                bucket_name="market_data",
                org="trading-system"
            )
            
        logger.info("InfluxDB initialized successfully")
        client.close()
        
    except Exception as e:
        logger.error(f"Error initializing InfluxDB: {e}")
        raise

def initialize_redis():
    """Initialize Redis"""
    try:
        config = load_config()
        redis_client = redis.Redis.from_url(config['system']['redis_url'])
        redis_client.ping()  # Test connection
        logger.info("Redis initialized successfully")
        redis_client.close()
        
    except Exception as e:
        logger.error(f"Error initializing Redis: {e}")
        raise

async def verify_gpu():
    """Verify GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            logger.info(f"GPU detected: {device.name}")
            logger.info(f"GPU memory: {device.total_memory / 1024**3:.2f}GB")
        else:
            logger.warning("No GPU detected. System will run in CPU-only mode")
            
    except Exception as e:
        logger.error(f"Error verifying GPU: {e}")
        raise

async def main():
    """Initialize all components"""
    try:
        # Create necessary directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Initialize components
        await verify_gpu()
        await initialize_postgres()
        await initialize_influxdb()
        initialize_redis()
        
        logger.info("System initialization completed successfully")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
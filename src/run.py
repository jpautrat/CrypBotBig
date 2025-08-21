import asyncio
import logging
from datetime import datetime
import argparse
from core.system_controller import SystemController
from core.config import load_config
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Elite Trading System")

async def start_system(args):
    """Initialize and start the trading system"""
    try:
        # Load configuration
        config = load_config()
        
        # Update config with command line arguments
        config['system'].update({
            'simulation_mode': not args.live,
            'debug_mode': args.debug
        })
        
        # Initialize system controller
        system = SystemController()
        await system.initialize()
        
        # Start the system
        if args.live:
            logger.info("Starting system in LIVE trading mode")
        else:
            logger.info("Starting system in SIMULATION mode")
            
        await system.start()
        
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        raise

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Elite Trading System")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--port", type=int, default=8000, help="Web interface port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Mount static files for web interface
    app.mount("/", StaticFiles(directory="web_ui", html=True), name="web_ui")
    
    # Start the system
    config = {
        "app": "src.run:app",
        "host": "0.0.0.0",
        "port": args.port,
        "reload": args.debug,
        "workers": 1,
        "loop": "asyncio"
    }
    
    # Start FastAPI server with system initialization
    asyncio.run(start_system(args))
    uvicorn.run(**config)
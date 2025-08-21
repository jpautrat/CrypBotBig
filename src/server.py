from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List
import numpy as np
import torch
import ccxt.async_support as ccxt
from core.config import load_config
from core.market_manager import MarketManager
from core.trade_executor import TradeExecutor
from core.risk_manager import RiskManager
from models.deep_learning import DeepLearningModel
from models.market_predictor import MarketPredictor
from strategies.multi_market import MultiMarketStrategy
from data.market_data import MarketDataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Elite Trading System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = load_config()
market_manager = MarketManager()
trade_executor = TradeExecutor()
risk_manager = RiskManager()
market_data = MarketDataManager()

# Initialize ML models with GPU support if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deep_learning_model = DeepLearningModel(device=device)
market_predictor = MarketPredictor(device=device)

# Active WebSocket connections
active_connections: Dict[int, WebSocket] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    try:
        await market_manager.initialize()
        await market_data.initialize()
        await deep_learning_model.load()
        logger.info("System initialized successfully")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time updates"""
    client_id = id(websocket)
    await websocket.accept()
    active_connections[client_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "subscribe":
                # Handle subscription to specific market data
                pass
            elif message["type"] == "trade":
                # Handle trade execution requests
                pass
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        del active_connections[client_id]

@app.get("/markets")
async def get_markets():
    """Get all available markets and their current status"""
    return await market_manager.get_all_markets()

@app.get("/portfolio")
async def get_portfolio():
    """Get current portfolio status and performance metrics"""
    return await trade_executor.get_portfolio_status()

@app.get("/analysis/{market_id}")
async def get_market_analysis(market_id: str):
    """Get detailed market analysis for a specific market"""
    return await market_predictor.analyze_market(market_id)

@app.post("/trade")
async def execute_trade(trade_params: dict, background_tasks: BackgroundTasks):
    """Execute a trade with the given parameters"""
    if await risk_manager.validate_trade(trade_params):
        background_tasks.add_task(trade_executor.execute_trade, trade_params)
        return {"status": "Trade scheduled for execution"}
    return {"status": "Trade rejected by risk manager"}

async def broadcast_updates():
    """Broadcast market updates to all connected clients"""
    while True:
        updates = await market_data.get_updates()
        for client in active_connections.values():
            try:
                await client.send_json(updates)
            except Exception as e:
                logger.error(f"Error broadcasting updates: {e}")
        await asyncio.sleep(0.1)  # 100ms update interval

if __name__ == "__main__":
    # Start the broadcast loop in the background
    asyncio.create_task(broadcast_updates())
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
import asyncio
import logging
from datetime import datetime
import torch
from typing import Dict, List, Optional

from core.system_controller import SystemController
from core.market_manager import MarketManager
from core.trade_executor import TradeExecutor
from core.risk_manager import RiskManager
from core.portfolio_manager import PortfolioManager
from models.deep_learning import DeepLearningModel
from models.market_predictor import MarketPredictor
from strategies.multi_market import MultiMarketStrategy
from strategies.arbitrage import ArbitrageDetector
from data.market_data import MarketDataManager
from data.database import DatabaseManager
from core.system_monitor import SystemMonitor
from core.config import load_config

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        # Load configuration
        self.config = load_config()
        
        # Initialize GPU device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.system_controller = None
        self.market_manager = None
        self.trade_executor = None
        self.risk_manager = None
        self.portfolio_manager = None
        self.deep_learning = None
        self.market_predictor = None
        self.strategy = None
        self.arbitrage = None
        self.market_data = None
        self.database = None
        self.monitor = None
        
    async def initialize(self):
        """Initialize all trading system components"""
        try:
            logger.info("Initializing trading system...")
            
            # Initialize core components
            self.system_controller = SystemController()
            self.market_manager = MarketManager()
            self.trade_executor = TradeExecutor()
            self.risk_manager = RiskManager()
            self.portfolio_manager = PortfolioManager(self.config)
            
            # Initialize ML models
            self.deep_learning = DeepLearningModel(device=self.device)
            self.market_predictor = MarketPredictor()
            
            # Initialize strategies
            self.strategy = MultiMarketStrategy()
            self.arbitrage = ArbitrageDetector(self.config)
            
            # Initialize data management
            self.market_data = MarketDataManager()
            self.database = DatabaseManager(self.config)
            
            # Initialize system monitor
            self.monitor = SystemMonitor(self.config)
            
            # Initialize each component
            await self._initialize_components()
            
            logger.info("Trading system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing trading system: {e}")
            raise
            
    async def _initialize_components(self):
        """Initialize each component in parallel"""
        try:
            init_tasks = [
                self.market_manager.initialize(),
                self.market_data.initialize(),
                self.database.initialize(),
                self.portfolio_manager.initialize(),
                self.system_controller.initialize()
            ]
            
            await asyncio.gather(*init_tasks)
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
            
    async def start(self):
        """Start the trading system"""
        try:
            logger.info("Starting trading system...")
            
            # Start monitoring
            asyncio.create_task(self.monitor.start_monitoring())
            
            # Start main processing loops
            await asyncio.gather(
                self._market_data_loop(),
                self._analysis_loop(),
                self._trading_loop(),
                self._arbitrage_loop(),
                self._model_update_loop()
            )
            
        except Exception as e:
            logger.error(f"Error starting trading system: {e}")
            raise
            
    async def _market_data_loop(self):
        """Process market data updates"""
        while True:
            try:
                # Get market updates from all exchanges
                for exchange in self.config['trading']['exchanges']:
                    pairs = self.config['trading']['trading_pairs'][exchange]
                    for pair in pairs:
                        market_data = await self.market_manager.get_market_depth(pair)
                        if market_data:
                            await self.market_data.process_market_data(pair, market_data)
                            
                await asyncio.sleep(0.01)  # 10ms interval
                
            except Exception as e:
                logger.error(f"Error in market data loop: {e}")
                await asyncio.sleep(1)
                
    async def _analysis_loop(self):
        """Analyze market conditions and generate predictions"""
        while True:
            try:
                # Get current market states
                market_states = await self.market_data.get_updates()
                
                # Generate predictions
                predictions = {}
                for symbol in market_states:
                    pred = await self.market_predictor.predict(
                        symbol,
                        market_states[symbol]
                    )
                    predictions[symbol] = pred
                    
                # Analyze opportunities
                opportunities = await self.strategy.analyze_opportunities(market_states)
                
                # Store analysis results
                await self.database.store_market_analysis({
                    'predictions': predictions,
                    'opportunities': opportunities,
                    'timestamp': datetime.now().timestamp()
                })
                
                await asyncio.sleep(0.1)  # 100ms interval
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(1)
                
    async def _trading_loop(self):
        """Execute trading decisions"""
        while True:
            try:
                # Get latest analysis
                market_analysis = await self.database.get_latest_analysis()
                
                if market_analysis:
                    # Validate opportunities with risk manager
                    for opportunity in market_analysis['opportunities']:
                        if await self.risk_manager.validate_trade(opportunity):
                            # Execute trade
                            await self.trade_executor.execute_trade(opportunity)
                            
                            # Update portfolio
                            await self.portfolio_manager.update_portfolio_value()
                            
                await asyncio.sleep(0.05)  # 50ms interval
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(1)
                
    async def _arbitrage_loop(self):
        """Look for arbitrage opportunities"""
        while True:
            try:
                # Get orderbook data
                orderbooks = await self.market_manager.get_all_orderbooks()
                
                # Find arbitrage opportunities
                opportunities = await self.arbitrage.analyze_opportunities(orderbooks)
                
                # Execute valid opportunities
                for opportunity in opportunities:
                    await self.arbitrage.execute_arbitrage(opportunity)
                    
                await asyncio.sleep(0.01)  # 10ms interval
                
            except Exception as e:
                logger.error(f"Error in arbitrage loop: {e}")
                await asyncio.sleep(1)
                
    async def _model_update_loop(self):
        """Update machine learning models"""
        while True:
            try:
                # Get training data
                training_window = self.config['ml']['training_window']
                market_data = await self.market_data.get_historical_data(
                    start_time=datetime.now(),
                    days=training_window
                )
                
                # Update models for each trading pair
                for exchange in self.config['trading']['exchanges']:
                    for pair in self.config['trading']['trading_pairs'][exchange]:
                        await self.market_predictor.train(pair, market_data)
                        
                # Wait for next update interval
                await asyncio.sleep(self.config['ml']['model_update_interval'])
                
            except Exception as e:
                logger.error(f"Error in model update loop: {e}")
                await asyncio.sleep(60)
                
    async def stop(self):
        """Stop the trading system"""
        try:
            logger.info("Stopping trading system...")
            
            # Cleanup resources
            self.market_manager.cleanup()
            self.market_data.cleanup()
            self.deep_learning.cleanup()
            self.market_predictor.cleanup()
            self.strategy.cleanup()
            await self.database.cleanup()
            
            logger.info("Trading system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")
            raise
            
    async def get_status(self) -> Dict:
        """Get current system status"""
        try:
            return {
                'system': await self.system_controller.get_system_status(),
                'portfolio': await self.portfolio_manager.get_portfolio_status(),
                'monitor': self.monitor.get_system_status()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/trading_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create and start bot
    bot = TradingBot()
    asyncio.run(bot.initialize())
    asyncio.run(bot.start())
import os
import json
from typing import Dict, Any
import base64
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self):
        self.config = {}
        self.encryption_key = None
        self._load_encryption_key()
        
    def _load_encryption_key(self):
        """Load or create encryption key for sensitive data"""
        key_file = ".key"
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                self.encryption_key = f.read()
        else:
            self.encryption_key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self.encryption_key)

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from environment and config files"""
        try:
            # Load environment variables
            load_dotenv()
            
            # System configuration
            self.config['system'] = {
                'max_threads': 32,  # Optimize for i7-11800H
                'gpu_enabled': True,  # Enable RTX 3070 support
                'gpu_memory_limit': 6144,  # 6GB GPU memory limit
                'cpu_threads': os.cpu_count(),
                'max_positions': 100,
                'risk_per_trade': 0.02,
                'database_url': os.getenv('DATABASE_URL', 'postgresql://localhost/trading'),
                'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
                'influxdb_url': os.getenv('INFLUXDB_URL', 'http://localhost:8086')
            }
            
            # Trading configuration
            self.config['trading'] = {
                'exchanges': ['kraken', 'binance', 'coinbase', 'kucoin'],
                'default_leverage': 1,
                'max_leverage': 3,
                'min_order_size': {
                    'BTC': 0.0001,
                    'ETH': 0.01,
                    'default': 0.1
                },
                'trading_pairs': self._load_trading_pairs(),
                'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
                'max_slippage': 0.001,
                'min_profit_threshold': 0.002,
                'max_drawdown': 0.25
            }
            
            # Risk management configuration
            self.config['risk'] = {
                'portfolio_stop_loss': 0.15,  # 15% maximum portfolio loss
                'position_stop_loss': 0.05,   # 5% maximum position loss
                'max_correlation': 0.7,       # Maximum correlation between positions
                'volatility_threshold': 0.05, # Maximum allowed volatility
                'max_exposure': 0.8,          # Maximum portfolio exposure
                'min_liquidity': 1000000     # Minimum daily volume in USD
            }
            
            # Machine learning configuration
            self.config['ml'] = {
                'model_update_interval': 3600,  # Update models hourly
                'training_window': 30,          # Training window in days
                'feature_columns': [
                    'price', 'volume', 'volatility',
                    'ema_5', 'ema_10', 'ema_20', 'ema_50',
                    'volume_ema', 'roc'
                ],
                'prediction_horizons': [1, 5, 15, 60],  # minutes
                'confidence_threshold': 0.75,
                'gpu_batch_size': 64
            }
            
            # Load API credentials
            self._load_api_credentials()
            
            logger.info("Configuration loaded successfully")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _load_trading_pairs(self) -> Dict[str, list]:
        """Load supported trading pairs for each exchange"""
        try:
            pairs = {
                'kraken': [
                    'BTC/USD', 'ETH/USD', 'XRP/USD', 'ADA/USD',
                    'DOT/USD', 'SOL/USD', 'DOGE/USD', 'MATIC/USD'
                ],
                'binance': [
                    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT',
                    'DOT/USDT', 'SOL/USDT', 'DOGE/USDT', 'MATIC/USDT'
                ],
                'coinbase': [
                    'BTC/USD', 'ETH/USD', 'XRP/USD', 'ADA/USD',
                    'DOT/USD', 'SOL/USD', 'DOGE/USD', 'MATIC/USD'
                ],
                'kucoin': [
                    'BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT',
                    'DOT/USDT', 'SOL/USDT', 'DOGE/USDT', 'MATIC/USDT'
                ]
            }
            return pairs
            
        except Exception as e:
            logger.error(f"Error loading trading pairs: {e}")
            return {}

    def _load_api_credentials(self):
        """Load and decrypt API credentials"""
        try:
            fernet = Fernet(self.encryption_key)
            
            # Initialize credentials dictionary
            self.config['credentials'] = {}
            
            # Load credentials for each exchange
            exchanges = ['kraken', 'binance', 'coinbase', 'kucoin']
            
            for exchange in exchanges:
                api_key = os.getenv(f'{exchange.upper()}_API_KEY')
                api_secret = os.getenv(f'{exchange.upper()}_API_SECRET')
                
                if api_key and api_secret:
                    # Encrypt credentials before storing in memory
                    encrypted_key = fernet.encrypt(api_key.encode())
                    encrypted_secret = fernet.encrypt(api_secret.encode())
                    
                    self.config['credentials'][exchange] = {
                        'api_key': encrypted_key,
                        'api_secret': encrypted_secret
                    }
                    
            logger.info("API credentials loaded and encrypted")
            
        except Exception as e:
            logger.error(f"Error loading API credentials: {e}")

    def get_api_credentials(self, exchange: str) -> Dict[str, str]:
        """Get decrypted API credentials for an exchange"""
        try:
            if exchange not in self.config['credentials']:
                raise ValueError(f"No credentials found for {exchange}")
                
            fernet = Fernet(self.encryption_key)
            credentials = self.config['credentials'][exchange]
            
            return {
                'api_key': fernet.decrypt(credentials['api_key']).decode(),
                'api_secret': fernet.decrypt(credentials['api_secret']).decode()
            }
            
        except Exception as e:
            logger.error(f"Error retrieving API credentials: {e}")
            return {}

    def update_config(self, section: str, key: str, value: Any):
        """Update configuration value"""
        try:
            if section not in self.config:
                self.config[section] = {}
            
            self.config[section][key] = value
            logger.info(f"Updated config: {section}.{key}")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")

    def save_config(self):
        """Save configuration to file"""
        try:
            # Remove sensitive data before saving
            config_to_save = self.config.copy()
            config_to_save.pop('credentials', None)
            
            with open('config.json', 'w') as f:
                json.dump(config_to_save, f, indent=4)
                
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

def load_config() -> Dict[str, Any]:
    """Helper function to load configuration"""
    config_manager = ConfigManager()
    return config_manager.load_config()
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
import pandas as pd

logger = logging.getLogger(__name__)

class TransformerPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output layers with skip connections
        self.fc1 = nn.Linear(d_model, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Take the last sequence output
        x = x[:, -1, :]
        
        # Output layers with residual connections
        identity = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x + self.fc1(identity)  # Skip connection
        
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MarketPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.scalers = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=32)
        self.batch_size = 64
        self.sequence_length = 100
        self.prediction_horizons = [1, 5, 15, 60]  # minutes
        
    async def initialize_model(self, symbol: str):
        """Initialize prediction model for a symbol"""
        try:
            model = TransformerPredictor().to(self.device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.0001,
                weight_decay=0.01
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2
            )
            
            self.models[symbol] = {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler
            }
            
            self.scalers[symbol] = StandardScaler()
            
            logger.info(f"Initialized prediction model for {symbol}")
            
        except Exception as e:
            logger.error(f"Error initializing model for {symbol}: {e}")
            
    async def prepare_features(
        self,
        market_data: pd.DataFrame,
        symbol: str
    ) -> torch.Tensor:
        """Prepare features for model input using GPU acceleration"""
        try:
            # Extract relevant features
            features = [
                'price', 'volume', 'volatility',
                'ema_5', 'ema_10', 'ema_20', 'ema_50',
                'volume_ema', 'roc'
            ]
            
            # Convert to numpy array
            feature_data = market_data[features].values
            
            # Scale features
            if symbol not in self.scalers:
                self.scalers[symbol] = StandardScaler()
                feature_data = self.scalers[symbol].fit_transform(feature_data)
            else:
                feature_data = self.scalers[symbol].transform(feature_data)
            
            # Convert to tensor and move to GPU
            feature_tensor = torch.FloatTensor(feature_data).to(self.device)
            
            # Create sequences
            sequences = []
            for i in range(len(feature_tensor) - self.sequence_length):
                sequence = feature_tensor[i:i + self.sequence_length]
                sequences.append(sequence)
            
            if sequences:
                return torch.stack(sequences)
            return None
            
        except Exception as e:
            logger.error(f"Error preparing features for {symbol}: {e}")
            return None

    async def train(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        epochs: int = 100
    ) -> Dict:
        """Train prediction model using GPU acceleration"""
        try:
            if symbol not in self.models:
                await self.initialize_model(symbol)
            
            model = self.models[symbol]['model']
            optimizer = self.models[symbol]['optimizer']
            scheduler = self.models[symbol]['scheduler']
            
            # Prepare training data
            X = await self.prepare_features(market_data, symbol)
            if X is None:
                return {'error': 'Failed to prepare features'}
            
            # Prepare targets (future prices)
            y = torch.FloatTensor(
                market_data['price'].values[self.sequence_length:]
            ).to(self.device)
            
            # Training metrics
            metrics = {
                'train_loss': [],
                'val_loss': []
            }
            
            # Split into train/val
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(epochs):
                # Training
                total_loss = 0
                for i in range(0, len(X_train), self.batch_size):
                    batch_X = X_train[i:i + self.batch_size]
                    batch_y = y_train[i:i + self.batch_size]
                    
                    optimizer.zero_grad()
                    predictions = model(batch_X)
                    loss = criterion(predictions.squeeze(), batch_y)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_predictions = model(X_val)
                    val_loss = criterion(val_predictions.squeeze(), y_val)
                
                # Update learning rate
                scheduler.step()
                
                # Record metrics
                metrics['train_loss'].append(total_loss / (len(X_train) / self.batch_size))
                metrics['val_loss'].append(val_loss.item())
                
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {metrics['train_loss'][-1]:.6f}, "
                        f"Val Loss: {metrics['val_loss'][-1]:.6f}"
                    )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {'error': str(e)}

    async def predict(
        self,
        symbol: str,
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Make price predictions using trained model"""
        try:
            if symbol not in self.models:
                return {'error': 'No model initialized for symbol'}
            
            model = self.models[symbol]['model']
            
            # Prepare features
            X = await self.prepare_features(market_data, symbol)
            if X is None:
                return {'error': 'Failed to prepare features'}
            
            # Make predictions
            model.eval()
            with torch.no_grad():
                predictions = model(X[-1].unsqueeze(0))
                
            # Scale predictions back to original range
            predictions = self.scalers[symbol].inverse_transform(
                predictions.cpu().numpy()
            )
            
            return {
                f'{horizon}m': float(pred)
                for horizon, pred in zip(self.prediction_horizons, predictions[0])
            }
            
        except Exception as e:
            logger.error(f"Error making predictions for {symbol}: {e}")
            return {'error': str(e)}

    async def analyze_market(self, symbol: str) -> Dict:
        """Perform comprehensive market analysis"""
        try:
            # Get predictions
            market_data = await self._get_market_data(symbol)
            predictions = await self.predict(symbol, market_data)
            
            # Calculate confidence scores
            confidence_scores = await self._calculate_confidence(
                symbol,
                market_data,
                predictions
            )
            
            # Analyze market regime
            regime = await self._detect_market_regime(market_data)
            
            return {
                'predictions': predictions,
                'confidence_scores': confidence_scores,
                'market_regime': regime,
                'timestamp': datetime.now().timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
            return {'error': str(e)}

    async def _get_market_data(self, symbol: str) -> pd.DataFrame:
        """Retrieve market data for analysis"""
        # Implementation will connect to market data manager
        pass

    async def _calculate_confidence(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        predictions: Dict
    ) -> Dict[str, float]:
        """Calculate confidence scores for predictions"""
        try:
            # Calculate prediction uncertainty
            model = self.models[symbol]['model']
            
            # Enable dropout for uncertainty estimation
            model.train()
            
            # Multiple forward passes for uncertainty estimation
            n_passes = 10
            predictions_list = []
            
            X = await self.prepare_features(market_data, symbol)
            if X is None:
                return {}
            
            with torch.no_grad():
                for _ in range(n_passes):
                    pred = model(X[-1].unsqueeze(0))
                    predictions_list.append(pred.cpu().numpy())
            
            # Calculate uncertainty metrics
            predictions_array = np.array(predictions_list)
            mean_predictions = np.mean(predictions_array, axis=0)
            std_predictions = np.std(predictions_array, axis=0)
            
            # Calculate confidence scores
            confidence_scores = {}
            for horizon, mean, std in zip(
                self.prediction_horizons,
                mean_predictions[0],
                std_predictions[0]
            ):
                # Higher confidence when std is lower
                confidence = 1.0 / (1.0 + std)
                confidence_scores[f'{horizon}m'] = float(confidence)
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {e}")
            return {}

    async def _detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime using statistical analysis"""
        try:
            # Calculate key metrics
            returns = np.diff(np.log(market_data['price'].values))
            volatility = np.std(returns) * np.sqrt(252)
            
            # Define regime thresholds
            if volatility < 0.1:
                regime = 'low_volatility'
            elif volatility < 0.25:
                regime = 'normal'
            else:
                regime = 'high_volatility'
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return 'unknown'

    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown()
        torch.cuda.empty_cache()
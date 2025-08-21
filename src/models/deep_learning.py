import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_length: int = 100):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]
        return sequence, target

class PricePredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3):
        super(PricePredictor, self).__init__()
        
        # LSTM layers for sequential data processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
        # Fully connected layers with residual connections
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(3)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_size, 1)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Apply attention mechanism
        attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output and apply fully connected layers with residual connections
        x = attention_out[:, -1, :]
        
        # Apply batch normalization
        x = self.batch_norm(x)
        
        # Residual connections through FC layers
        for fc in self.fc_layers:
            residual = x
            x = fc(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = x + residual
        
        # Final output
        return self.output(x)

class DeepLearningModel:
    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, PricePredictor] = {}
        self.optimizers: Dict[str, optim.Adam] = {}
        self.schedulers: Dict[str, optim.lr_scheduler.ReduceLROnPlateau] = {}
        
    async def initialize_model(self, pair: str, input_size: int):
        """Initialize a new model for a trading pair"""
        model = PricePredictor(input_size=input_size).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.models[pair] = model
        self.optimizers[pair] = optimizer
        self.schedulers[pair] = scheduler
        
        logger.info(f"Initialized model for {pair} on {self.device}")

    async def train(self, pair: str, features: np.ndarray, targets: np.ndarray,
                   batch_size: int = 64, epochs: int = 10) -> Dict[str, float]:
        """Train the model for a specific trading pair"""
        if pair not in self.models:
            await self.initialize_model(pair, features.shape[-1])
        
        model = self.models[pair]
        optimizer = self.optimizers[pair]
        scheduler = self.schedulers[pair]
        
        # Prepare data
        dataset = TimeSeriesDataset(features)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        
        # Training metrics
        metrics = {
            'train_loss': [],
            'val_loss': []
        }
        
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_features, batch_targets in dataloader:
                # Move data to GPU if available
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Update learning rate
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            metrics['train_loss'].append(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        return metrics

    async def predict(self, pair: str, features: np.ndarray) -> np.ndarray:
        """Make predictions for a trading pair"""
        if pair not in self.models:
            raise ValueError(f"No model initialized for {pair}")
        
        model = self.models[pair]
        model.eval()
        
        with torch.no_grad():
            # Prepare input data
            x = torch.FloatTensor(features).to(self.device)
            
            # Make prediction
            outputs = model(x)
            predictions = outputs.cpu().numpy()
        
        return predictions

    async def save_model(self, pair: str, path: str):
        """Save model state"""
        if pair in self.models:
            state = {
                'model_state': self.models[pair].state_dict(),
                'optimizer_state': self.optimizers[pair].state_dict(),
                'scheduler_state': self.schedulers[pair].state_dict()
            }
            torch.save(state, f"{path}/{pair}_model.pth")
            logger.info(f"Saved model state for {pair}")

    async def load_model(self, pair: str, path: str):
        """Load model state"""
        try:
            state = torch.load(f"{path}/{pair}_model.pth")
            await self.initialize_model(pair, state['model_state']['lstm.weight_ih_l0'].shape[1])
            
            self.models[pair].load_state_dict(state['model_state'])
            self.optimizers[pair].load_state_dict(state['optimizer_state'])
            self.schedulers[pair].load_state_dict(state['scheduler_state'])
            
            logger.info(f"Loaded model state for {pair}")
        except Exception as e:
            logger.error(f"Error loading model for {pair}: {e}")

    def cleanup(self):
        """Free GPU memory"""
        for model in self.models.values():
            model.cpu()
        torch.cuda.empty_cache()

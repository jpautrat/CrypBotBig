import numpy as np
import torch
from typing import Dict, List, Optional
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thread_pool = ThreadPoolExecutor(max_workers=32)
        
        # Portfolio state
        self.positions = {}
        self.open_orders = {}
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_value = 0.0
        self.cash_balance = {}
        self.risk_exposure = 0.0
        
        # Risk limits
        self.max_position_size = 0.2  # 20% of portfolio
        self.max_total_risk = 0.15    # 15% portfolio risk
        self.position_stop_loss = 0.05 # 5% position stop loss
        
    async def initialize(self):
        """Initialize portfolio state"""
        try:
            # Load initial balances from exchanges
            for exchange in self.config['trading']['exchanges']:
                await self._fetch_exchange_balance(exchange)
                
            # Calculate initial portfolio value
            await self.update_portfolio_value()
            
            logger.info("Portfolio manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing portfolio manager: {e}")
            raise
            
    async def update_portfolio_value(self):
        """Update total portfolio value using GPU acceleration"""
        try:
            # Gather all position values
            position_values = []
            for symbol, position in self.positions.items():
                current_price = await self._get_current_price(symbol)
                if current_price:
                    value = position['amount'] * current_price
                    position_values.append(value)
                    
            # Convert to tensor for GPU calculation
            if position_values:
                values_tensor = torch.tensor(position_values, device=self.device)
                self.total_value = values_tensor.sum().item()
                
                # Add cash balances
                for currency, amount in self.cash_balance.items():
                    if currency == 'USD' or currency == 'USDT':
                        self.total_value += amount
                        
            # Calculate unrealized P&L
            self._calculate_unrealized_pnl()
            
            # Update risk exposure
            self._update_risk_exposure()
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
            
    async def open_position(self, trade_params: Dict) -> bool:
        """Open a new position"""
        try:
            symbol = trade_params['symbol']
            amount = trade_params['amount']
            price = trade_params['price']
            side = trade_params['side']
            
            # Validate position size
            if not await self._validate_position_size(amount * price):
                return False
                
            # Check risk limits
            if not await self._check_risk_limits(trade_params):
                return False
                
            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'amount': amount if side == 'buy' else -amount,
                    'avg_price': price,
                    'unrealized_pnl': 0.0,
                    'stop_loss': price * (1 - self.position_stop_loss if side == 'buy' else 1 + self.position_stop_loss)
                }
            else:
                # Update existing position
                current_amount = self.positions[symbol]['amount']
                current_price = self.positions[symbol]['avg_price']
                
                # Calculate new average price
                total_amount = current_amount + (amount if side == 'buy' else -amount)
                if total_amount != 0:
                    avg_price = (current_amount * current_price + amount * price) / total_amount
                    self.positions[symbol]['avg_price'] = avg_price
                    self.positions[symbol]['amount'] = total_amount
                    
            # Update portfolio value
            await self.update_portfolio_value()
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False
            
    async def close_position(self, symbol: str, amount: float = None) -> bool:
        """Close a position partially or completely"""
        try:
            if symbol not in self.positions:
                return False
                
            position = self.positions[symbol]
            
            # Determine closing amount
            close_amount = amount if amount else abs(position['amount'])
            if close_amount > abs(position['amount']):
                return False
                
            # Get current price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return False
                
            # Calculate realized P&L
            avg_price = position['avg_price']
            pnl = (current_price - avg_price) * close_amount
            if position['amount'] < 0:  # Short position
                pnl = -pnl
                
            self.realized_pnl += pnl
            
            # Update position
            remaining_amount = position['amount'] - (close_amount if position['amount'] > 0 else -close_amount)
            if abs(remaining_amount) < 1e-8:
                del self.positions[symbol]
            else:
                position['amount'] = remaining_amount
                
            # Update portfolio value
            await self.update_portfolio_value()
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
            
    async def _validate_position_size(self, position_value: float) -> bool:
        """Validate if position size is within limits"""
        try:
            # Check against maximum position size
            max_position_value = self.total_value * self.max_position_size
            return position_value <= max_position_value
            
        except Exception as e:
            logger.error(f"Error validating position size: {e}")
            return False
            
    async def _check_risk_limits(self, trade_params: Dict) -> bool:
        """Check if trade is within risk limits"""
        try:
            # Calculate potential new risk exposure
            new_risk = self._calculate_position_risk(trade_params)
            total_risk = self.risk_exposure + new_risk
            
            # Check against maximum risk
            return total_risk <= self.max_total_risk * self.total_value
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
            
    def _calculate_position_risk(self, trade_params: Dict) -> float:
        """Calculate risk for a potential position using GPU"""
        try:
            # Convert parameters to tensors
            price = torch.tensor(trade_params['price'], device=self.device)
            amount = torch.tensor(trade_params['amount'], device=self.device)
            volatility = torch.tensor(trade_params.get('volatility', 0.02), device=self.device)
            
            # Calculate Value at Risk (VaR)
            position_value = price * amount
            var = position_value * volatility * torch.sqrt(torch.tensor(252.0, device=self.device))
            
            return var.item()
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return float('inf')
            
    def _calculate_unrealized_pnl(self):
        """Calculate unrealized P&L for all positions"""
        try:
            total_pnl = 0.0
            for symbol, position in self.positions.items():
                current_price = await self._get_current_price(symbol)
                if current_price:
                    pnl = (current_price - position['avg_price']) * position['amount']
                    position['unrealized_pnl'] = pnl
                    total_pnl += pnl
                    
            self.unrealized_pnl = total_pnl
            
        except Exception as e:
            logger.error(f"Error calculating unrealized P&L: {e}")
            
    def _update_risk_exposure(self):
        """Update portfolio risk exposure"""
        try:
            total_risk = 0.0
            for symbol, position in self.positions.items():
                risk = self._calculate_position_risk({
                    'symbol': symbol,
                    'price': position['avg_price'],
                    'amount': position['amount']
                })
                total_risk += risk
                
            self.risk_exposure = total_risk
            
        except Exception as e:
            logger.error(f"Error updating risk exposure: {e}")
            
    async def _fetch_exchange_balance(self, exchange: str):
        """Fetch balance from an exchange"""
        # Implementation will connect to exchange manager
        pass
        
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        # Implementation will connect to market data manager
        pass
        
    async def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        try:
            return {
                'total_value': self.total_value,
                'realized_pnl': self.realized_pnl,
                'unrealized_pnl': self.unrealized_pnl,
                'risk_exposure': self.risk_exposure,
                'positions': self.positions,
                'cash_balance': self.cash_balance,
                'timestamp': datetime.now().timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {}
            
    def cleanup(self):
        """Cleanup resources"""
        self.thread_pool.shutdown()
        torch.cuda.empty_cache()
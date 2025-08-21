import psutil
import torch
import logging
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import numpy as np
import GPUtil
from collections import deque

logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self, config: Dict):
        self.config = config
        self.metrics_history = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'gpu_usage': deque(maxlen=1000),
            'gpu_memory': deque(maxlen=1000),
            'trade_latency': deque(maxlen=1000),
            'prediction_time': deque(maxlen=1000),
            'api_latency': deque(maxlen=1000)
        }
        
        # Performance thresholds
        self.cpu_threshold = 90  # 90% CPU usage
        self.memory_threshold = 85  # 85% RAM usage
        self.gpu_memory_threshold = 90  # 90% GPU memory
        self.latency_threshold = 100  # 100ms
        
        # System state
        self.is_healthy = True
        self.warnings = []
        self.errors = []
        
        # Initialize GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()
            self.gpu_properties = torch.cuda.get_device_properties(self.gpu_device)
            
    async def start_monitoring(self):
        """Start system monitoring"""
        try:
            while True:
                await self.check_system_health()
                await asyncio.sleep(1)  # Check every second
                
        except Exception as e:
            logger.error(f"Error in system monitoring: {e}")
            
    async def check_system_health(self):
        """Check overall system health"""
        try:
            # Collect current metrics
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            
            # Store metrics
            self.metrics_history['cpu_usage'].append(cpu_usage)
            self.metrics_history['memory_usage'].append(memory_usage)
            
            # Check GPU metrics if available
            if self.gpu_available:
                gpu_metrics = self.get_gpu_metrics()
                self.metrics_history['gpu_usage'].append(gpu_metrics['gpu_usage'])
                self.metrics_history['gpu_memory'].append(gpu_metrics['memory_usage'])
                
            # Check for warning conditions
            self.warnings = []
            if cpu_usage > self.cpu_threshold:
                self.warnings.append(f"High CPU usage: {cpu_usage}%")
            if memory_usage > self.memory_threshold:
                self.warnings.append(f"High memory usage: {memory_usage}%")
            
            if self.gpu_available:
                if gpu_metrics['memory_usage'] > self.gpu_memory_threshold:
                    self.warnings.append(f"High GPU memory usage: {gpu_metrics['memory_usage']}%")
                    
            # Calculate performance metrics
            perf_metrics = self.calculate_performance_metrics()
            
            # Update system health status
            self.is_healthy = len(self.warnings) == 0 and len(self.errors) == 0
            
            # Log status
            if not self.is_healthy:
                logger.warning("System health issues detected:")
                for warning in self.warnings:
                    logger.warning(f"- {warning}")
                for error in self.errors:
                    logger.error(f"- {error}")
                    
            return {
                'is_healthy': self.is_healthy,
                'warnings': self.warnings,
                'errors': self.errors,
                'metrics': perf_metrics
            }
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return None
            
    def get_gpu_metrics(self) -> Dict:
        """Get GPU performance metrics"""
        try:
            if not self.gpu_available:
                return {'gpu_usage': 0, 'memory_usage': 0}
                
            # Get NVIDIA GPU metrics using GPUtil
            gpu = GPUtil.getGPUs()[self.gpu_device]
            
            return {
                'gpu_usage': gpu.load * 100,
                'memory_usage': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'temperature': gpu.temperature,
                'power_usage': gpu.powerUsage if hasattr(gpu, 'powerUsage') else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return {'gpu_usage': 0, 'memory_usage': 0}
            
    def calculate_performance_metrics(self) -> Dict:
        """Calculate system performance metrics"""
        try:
            metrics = {}
            
            # Calculate average metrics
            for metric_name, values in self.metrics_history.items():
                if values:
                    metrics[f'avg_{metric_name}'] = np.mean(values)
                    metrics[f'max_{metric_name}'] = np.max(values)
                    metrics[f'min_{metric_name}'] = np.min(values)
                    
            # Calculate trade execution statistics
            if self.metrics_history['trade_latency']:
                trade_latencies = np.array(self.metrics_history['trade_latency'])
                metrics['trade_latency_99th'] = np.percentile(trade_latencies, 99)
                metrics['trade_latency_95th'] = np.percentile(trade_latencies, 95)
                metrics['trade_latency_mean'] = np.mean(trade_latencies)
                
            # Calculate prediction performance
            if self.metrics_history['prediction_time']:
                pred_times = np.array(self.metrics_history['prediction_time'])
                metrics['prediction_time_mean'] = np.mean(pred_times)
                metrics['predictions_per_second'] = 1000 / metrics['prediction_time_mean']
                
            # Add current resource usage
            metrics.update({
                'cpu_usage': psutil.cpu_percent(interval=None),
                'memory_usage': psutil.virtual_memory().percent,
                'swap_usage': psutil.swap_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            })
            
            # Add GPU metrics if available
            if self.gpu_available:
                gpu_metrics = self.get_gpu_metrics()
                metrics.update({
                    'gpu_usage': gpu_metrics['gpu_usage'],
                    'gpu_memory_usage': gpu_metrics['memory_usage'],
                    'gpu_temperature': gpu_metrics['temperature'],
                    'gpu_power_usage': gpu_metrics['power_usage']
                })
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
            
    def log_trade_latency(self, latency_ms: float):
        """Log trade execution latency"""
        try:
            self.metrics_history['trade_latency'].append(latency_ms)
            
            # Check for latency issues
            if latency_ms > self.latency_threshold:
                logger.warning(f"High trade latency detected: {latency_ms}ms")
                
        except Exception as e:
            logger.error(f"Error logging trade latency: {e}")
            
    def log_prediction_time(self, prediction_time_ms: float):
        """Log model prediction time"""
        try:
            self.metrics_history['prediction_time'].append(prediction_time_ms)
            
        except Exception as e:
            logger.error(f"Error logging prediction time: {e}")
            
    def log_api_latency(self, latency_ms: float):
        """Log API request latency"""
        try:
            self.metrics_history['api_latency'].append(latency_ms)
            
        except Exception as e:
            logger.error(f"Error logging API latency: {e}")
            
    async def optimize_performance(self):
        """Optimize system performance based on monitoring data"""
        try:
            # Check CPU usage
            if np.mean(list(self.metrics_history['cpu_usage'])) > self.cpu_threshold:
                logger.info("High CPU usage detected, optimizing workload...")
                await self._optimize_cpu_usage()
                
            # Check memory usage
            if np.mean(list(self.metrics_history['memory_usage'])) > self.memory_threshold:
                logger.info("High memory usage detected, cleaning memory...")
                await self._optimize_memory_usage()
                
            # Check GPU memory if available
            if self.gpu_available:
                gpu_memory = np.mean(list(self.metrics_history['gpu_memory']))
                if gpu_memory > self.gpu_memory_threshold:
                    logger.info("High GPU memory usage detected, optimizing...")
                    await self._optimize_gpu_usage()
                    
        except Exception as e:
            logger.error(f"Error optimizing performance: {e}")
            
    async def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        try:
            # Implement CPU optimization strategies
            pass
            
        except Exception as e:
            logger.error(f"Error optimizing CPU usage: {e}")
            
    async def _optimize_memory_usage(self):
        """Optimize memory usage"""
        try:
            # Clear Python memory
            import gc
            gc.collect()
            
            # Clear GPU memory if available
            if self.gpu_available:
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error optimizing memory usage: {e}")
            
    async def _optimize_gpu_usage(self):
        """Optimize GPU usage"""
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Implement GPU optimization strategies
            pass
            
        except Exception as e:
            logger.error(f"Error optimizing GPU usage: {e}")
            
    def get_system_status(self) -> Dict:
        """Get current system status"""
        try:
            return {
                'is_healthy': self.is_healthy,
                'warnings': self.warnings,
                'errors': self.errors,
                'metrics': self.calculate_performance_metrics(),
                'timestamp': datetime.now().timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}
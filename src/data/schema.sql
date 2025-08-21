-- Market Data Schema
CREATE TABLE market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);

-- Trade History
CREATE TABLE trades (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    amount DECIMAL(20,8) NOT NULL,
    cost DECIMAL(20,8) NOT NULL,
    fee DECIMAL(20,8) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    order_id VARCHAR(100),
    strategy VARCHAR(50),
    prediction_confidence DECIMAL(10,8),
    market_score DECIMAL(10,8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, timestamp);

-- Portfolio Snapshots
CREATE TABLE portfolio_snapshots (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    total_value_usd DECIMAL(20,8) NOT NULL,
    realized_pnl DECIMAL(20,8) NOT NULL,
    unrealized_pnl DECIMAL(20,8) NOT NULL,
    risk_exposure DECIMAL(10,8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_portfolio_snapshots_timestamp ON portfolio_snapshots(timestamp);

-- Model Predictions
CREATE TABLE model_predictions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    predicted_price DECIMAL(20,8) NOT NULL,
    confidence_score DECIMAL(10,8) NOT NULL,
    actual_price DECIMAL(20,8),
    error_margin DECIMAL(20,8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_predictions_symbol_timestamp ON model_predictions(symbol, timestamp);

-- Market Analysis
CREATE TABLE market_analysis (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    volatility DECIMAL(10,8) NOT NULL,
    trend_strength DECIMAL(10,8) NOT NULL,
    market_regime VARCHAR(20) NOT NULL,
    trading_signal VARCHAR(20),
    correlation_score DECIMAL(10,8),
    volume_profile DECIMAL(10,8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_market_analysis_symbol_timestamp ON market_analysis(symbol, timestamp);

-- System Performance Metrics
CREATE TABLE system_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    cpu_usage DECIMAL(10,8) NOT NULL,
    memory_usage DECIMAL(10,8) NOT NULL,
    gpu_usage DECIMAL(10,8) NOT NULL,
    gpu_memory_usage DECIMAL(10,8) NOT NULL,
    api_latency_ms INTEGER NOT NULL,
    active_models INTEGER NOT NULL,
    prediction_time_ms INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Error Logs
CREATE TABLE error_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    component VARCHAR(50) NOT NULL,
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    resolution_status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_error_logs_timestamp ON error_logs(timestamp);

-- Trading Strategies
CREATE TABLE strategies (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    description TEXT,
    parameters JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Strategy Performance
CREATE TABLE strategy_performance (
    id BIGSERIAL PRIMARY KEY,
    strategy_id BIGINT REFERENCES strategies(id),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    win_rate DECIMAL(10,8) NOT NULL,
    profit_factor DECIMAL(10,8) NOT NULL,
    sharpe_ratio DECIMAL(10,8) NOT NULL,
    max_drawdown DECIMAL(10,8) NOT NULL,
    total_trades INTEGER NOT NULL,
    profitable_trades INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_strategy_performance_strategy_timestamp ON strategy_performance(strategy_id, timestamp);
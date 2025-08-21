// WebSocket connection
let ws;
const chartData = {};
const charts = {};

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeWebSocket();
    initializeCharts();
    setupEventListeners();
    initializeExchangeOptions();
});

function initializeWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateUI(data);
    };
    
    ws.onclose = () => {
        console.log('WebSocket connection closed. Reconnecting...');
        setTimeout(initializeWebSocket, 1000);
    };
}

function initializeCharts() {
    // Initialize TradingView widget
    new TradingView.widget({
        "width": '100%',
        "height": 600,
        "symbol": "KRAKEN:BTCUSD",
        "interval": "1",
        "timezone": "exchange",
        "theme": "dark",
        "style": "1",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview-widget"
    });
    
    // Initialize performance charts
    const volCtx = document.getElementById('volatility-chart').getContext('2d');
    charts.volatility = new Chart(volCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Volatility',
                data: [],
                borderColor: '#4caf50',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
    
    // Initialize correlation matrix
    const corrCtx = document.getElementById('correlation-matrix');
    const corrData = {
        type: 'heatmap',
        x: [],
        y: [],
        z: [],
        colorscale: 'Viridis'
    };
    Plotly.newPlot(corrCtx, [corrData], {
        title: 'Asset Correlation Matrix'
    });
}

function setupEventListeners() {
    // Buy button
    document.getElementById('buy-btn').addEventListener('click', () => {
        const params = getTradeParams();
        if (validateTradeParams(params)) {
            executeTrade({ ...params, side: 'buy' });
        }
    });
    
    // Sell button
    document.getElementById('sell-btn').addEventListener('click', () => {
        const params = getTradeParams();
        if (validateTradeParams(params)) {
            executeTrade({ ...params, side: 'sell' });
        }
    });
    
    // Exchange selection
    document.getElementById('exchange-select').addEventListener('change', (e) => {
        updateTradingPairs(e.target.value);
    });
}

function getTradeParams() {
    return {
        symbol: document.getElementById('pair-select').value,
        exchange: document.getElementById('exchange-select').value,
        type: document.getElementById('order-type').value,
        amount: parseFloat(document.getElementById('position-size').value)
    };
}

function validateTradeParams(params) {
    if (!params.symbol) {
        alert('Please select a trading pair');
        return false;
    }
    if (!params.exchange) {
        alert('Please select an exchange');
        return false;
    }
    if (!params.amount || params.amount <= 0) {
        alert('Please enter a valid position size');
        return false;
    }
    return true;
}

async function executeTrade(params) {
    try {
        const response = await fetch('/trade', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        
        const result = await response.json();
        if (result.status === 'Trade scheduled for execution') {
            alert('Trade executed successfully');
        } else {
            alert(`Trade failed: ${result.status}`);
        }
    } catch (error) {
        console.error('Error executing trade:', error);
        alert('Error executing trade. Please try again.');
    }
}

function updateUI(data) {
    // Update portfolio value
    if (data.portfolio) {
        document.getElementById('portfolio-value').textContent = 
            `Portfolio: $${data.portfolio.total_value.toFixed(2)}`;
    }
    
    // Update active trades
    if (data.trading_signals) {
        document.getElementById('active-trades').textContent = 
            `Active Trades: ${Object.keys(data.trading_signals).length}`;
    }
    
    // Update performance metrics
    if (data.metrics) {
        updatePerformanceMetrics(data.metrics);
    }
    
    // Update positions table
    if (data.portfolio && data.portfolio.positions) {
        updatePositionsTable(data.portfolio.positions);
    }
    
    // Update charts
    if (data.market_analysis) {
        updateCharts(data.market_analysis);
    }
}

function updatePerformanceMetrics(metrics) {
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML = `
        <div class="performance-metric">
            <strong>Win Rate:</strong> ${(metrics.win_rate * 100).toFixed(2)}%
        </div>
        <div class="performance-metric">
            <strong>Profit Factor:</strong> ${metrics.profit_factor.toFixed(2)}
        </div>
        <div class="performance-metric">
            <strong>Sharpe Ratio:</strong> ${metrics.sharpe_ratio.toFixed(2)}
        </div>
        <div class="performance-metric">
            <strong>Max Drawdown:</strong> ${(metrics.max_drawdown * 100).toFixed(2)}%
        </div>
    `;
}

function updatePositionsTable(positions) {
    const table = document.getElementById('positions-table');
    let html = `
        <table class="table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Size</th>
                    <th>Entry Price</th>
                    <th>Current Price</th>
                    <th>P&L</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    for (const position of positions) {
        const pnlClass = position.unrealized_pnl >= 0 ? 'metric-positive' : 'metric-negative';
        html += `
            <tr>
                <td>${position.symbol}</td>
                <td>${position.amount.toFixed(8)}</td>
                <td>$${position.avg_price.toFixed(2)}</td>
                <td>$${position.current_price.toFixed(2)}</td>
                <td class="${pnlClass}">$${position.unrealized_pnl.toFixed(2)}</td>
            </tr>
        `;
    }
    
    html += '</tbody></table>';
    table.innerHTML = html;
}

function updateCharts(analysis) {
    // Update volatility chart
    if (charts.volatility && analysis.volatility_history) {
        charts.volatility.data.labels = analysis.volatility_history.map(v => v.timestamp);
        charts.volatility.data.datasets[0].data = analysis.volatility_history.map(v => v.value);
        charts.volatility.update();
    }
    
    // Update correlation matrix
    if (analysis.correlation_matrix) {
        const corrCtx = document.getElementById('correlation-matrix');
        const update = {
            x: analysis.correlation_matrix.symbols,
            y: analysis.correlation_matrix.symbols,
            z: analysis.correlation_matrix.data
        };
        Plotly.update(corrCtx, update);
    }
    
    // Update sentiment analysis
    if (analysis.market_sentiment) {
        const sentimentCtx = document.getElementById('sentiment-analysis');
        const sentimentData = {
            values: Object.values(analysis.market_sentiment),
            labels: Object.keys(analysis.market_sentiment),
            type: 'pie'
        };
        Plotly.newPlot(sentimentCtx, [sentimentData]);
    }
}

async function initializeExchangeOptions() {
    try {
        const response = await fetch('/markets');
        const markets = await response.json();
        
        const exchangeSelect = document.getElementById('exchange-select');
        exchangeSelect.innerHTML = '<option value="">Select Exchange</option>';
        
        for (const exchange of markets.exchanges) {
            exchangeSelect.innerHTML += `
                <option value="${exchange}">${exchange.toUpperCase()}</option>
            `;
        }
    } catch (error) {
        console.error('Error loading exchanges:', error);
    }
}

async function updateTradingPairs(exchange) {
    try {
        const response = await fetch(`/markets/${exchange}`);
        const pairs = await response.json();
        
        const pairSelect = document.getElementById('pair-select');
        pairSelect.innerHTML = '<option value="">Select Pair</option>';
        
        for (const pair of pairs) {
            pairSelect.innerHTML += `
                <option value="${pair}">${pair}</option>
            `;
        }
    } catch (error) {
        console.error('Error loading trading pairs:', error);
    }
}
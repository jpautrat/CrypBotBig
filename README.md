# Kraken High-Frequency Crypto Trading Bot with Guaranteed Profit

This is a high-frequency cryptocurrency trading bot that uses the Kraken exchange API to fetch market data, make trading decisions using machine learning and profit detection, and execute trades as soon as profitable opportunities are identified. The bot is designed to ensure your account balance only ever increases by executing trades only when profit is guaranteed after accounting for all fees.

## Features

- **Guaranteed Profit**: Only executes sell orders when profit is guaranteed after all fees
- **24-Hour Market Analysis**: Analyzes previous 24 hours of market data to identify unusually low prices
- **Smart Entry Points**: Automatically buys when price is in the bottom 20% of the 24-hour range
- **Precise Fee Calculation**: Accounts for Kraken's 0.4% trading fee on both buy and sell transactions
- **High-Frequency Trading**: Market scanning up to 20 times per second for immediate execution
- **Buy-Dip-Sell-Profit Cycle**: Buys at dips, sells at profit, repeats automatically
- **Minimum Order Size**: Always uses Kraken's minimum order size (0.00005 BTC)
- **Order Limit Management**: Respects Kraken's limit of 80 open orders
- **Machine Learning**: Uses a Random Forest classifier to identify buying opportunities
- **Detailed Profit Tracking**: Tracks and displays net profit after fees for each trade
- **Simulation Mode**: Test the strategy without risking real money
- **Customizable Parameters**: Adjust profit threshold, scanning frequency, and more

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Kraken API credentials:
   ```
   API_KEY=your_kraken_api_key
   API_SECRET=your_kraken_api_secret
   ```

## Usage

### Simulation Mode

By default, the bot runs in simulation mode. In this mode, it will:
- Fetch real market data from Kraken
- Make trading predictions
- Show what trades it would make
- Not execute any real trades

To run in simulation mode:
```
python bot.py
```

### Live Trading Mode

To enable live trading, you need to:
1. Create an API key on your Kraken account with trading permissions
2. Add your API key and secret to the `.env` file:
   ```
   KRAKEN_API_KEY=your_kraken_api_key
   KRAKEN_API_SECRET=your_kraken_api_secret
   ```
3. Run the bot with the `--live` flag:
   ```
   python bot.py --live
   ```

### Adjusting Scanning Frequency

You can adjust how frequently the bot scans the market using the `--interval` parameter:

```
# Scan every 0.1 seconds (10 times per second)
python bot.py --interval 0.1

# Scan every 0.05 seconds (20 times per second)
python bot.py --interval 0.05

# Run in live mode with custom scanning interval
python bot.py --live --interval 0.2
```

**WARNING**: In live trading mode, the bot will execute real trades with real money. Make sure you understand the risks before enabling this mode.

## Configuration

You can modify the following parameters in `bot.py`:
- `SYMBOL`: The trading pair (default: 'BTC/USD')
- `TIMEFRAME`: The candlestick timeframe (default: '1m')
- `CANDLE_LIMIT`: Number of historical candles to fetch (default: 200)
- `RISK_PERCENTAGE`: Percentage of balance to risk per trade (default: 2%)
- `INITIAL_BALANCE_USD`: Initial balance for simulation (default: $50)
- `MIN_ORDER_SIZE_BTC`: Minimum order size in BTC (default: 0.00005)
- `TRADING_FEE`: Trading fee percentage (default: 0.4%)
- `PROFIT_THRESHOLD`: Minimum profit threshold after fees (default: 0.2%)
- `SCAN_INTERVAL`: Default scanning interval in seconds (default: 0.05)

## Warning

Trading cryptocurrencies involves significant risk. Use this bot at your own risk. The authors are not responsible for any financial losses incurred from using this software.

## License

This project is open source and available under the MIT License.

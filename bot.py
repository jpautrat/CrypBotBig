import ccxt, os, time
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Load secure API credentials
load_dotenv()

# Initialize Kraken trading API
# For simulation mode, we don't need API credentials
exchange = ccxt.kraken()

# If API credentials are available, use them for real trading
api_key = os.getenv('KRAKEN_API_KEY') or os.getenv('API_KEY')
api_secret = os.getenv('KRAKEN_API_SECRET') or os.getenv('API_SECRET')
if api_key and api_secret:
    exchange = ccxt.kraken({
        'apiKey': api_key,
        'secret': api_secret,
    })
    print("API credentials found. Real trading is possible if credentials are valid.")
else:
    print("No API credentials found. Running in simulation mode only.")

# Bot configuration
SYMBOL = 'BTC/USD'  # Kraken uses BTC/USD for Bitcoin
TIMEFRAME = '1m'  # Use 1-minute candles for more frequent updates
CANDLE_LIMIT = 200
RISK_PERCENTAGE = 0.02  # Risk 2% per trade
INITIAL_BALANCE_USD = 50
MIN_ORDER_SIZE_BTC = 0.00005  # Minimum order size in BTC (Kraken requirement)
MAX_OPEN_ORDERS = 80  # Maximum number of open orders allowed on Kraken
MAX_SCHEDULED_ORDERS = 25  # Maximum number of scheduled orders allowed on Kraken
TRADING_FEE = 0.004  # 0.4% trading fee on Kraken
SCAN_INTERVAL = 0.05  # Scan market every 0.05 seconds (20 times per second)
PROFIT_THRESHOLD = 0.002  # 0.2% minimum profit threshold (after fees)
SCALER = StandardScaler()

# Cache for historical data to reduce API calls
last_fetch_time = 0
cached_candles = None

def fetch_historical_data(verbose=True):
    global last_fetch_time, cached_candles

    current_time = time.time()

    # Only fetch new data if it's been more than 10 seconds since last fetch
    # or if we don't have any cached data
    if cached_candles is None or (current_time - last_fetch_time) > 10:
        if verbose:
            print(f"Fetching historical data for {SYMBOL}...")

        try:
            candles = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=CANDLE_LIMIT)
            if verbose:
                print(f"Received {len(candles)} candles.")

            cached_candles = candles
            last_fetch_time = current_time
        except Exception as e:
            print(f"Error fetching OHLCV data: {e}")
            # If we have cached data, use it rather than failing
            if cached_candles is None:
                raise
            print("Using cached data instead.")
    elif verbose:
        print(f"Using cached data ({len(cached_candles)} candles).")

    # Create DataFrame from candles
    df = pd.DataFrame(cached_candles, columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

    return df

def generate_features(df):
    df['return'] = df['Close'].pct_change()
    df['ma_fast'] = df['Close'].rolling(window=5).mean()
    df['ma_slow'] = df['Close'].rolling(window=10).mean()
    df['signal'] = np.where(df['ma_fast'] > df['ma_slow'], 1, 0)
    df.dropna(inplace=True)
    return df

def train_ml_model(df):
    features = ['return', 'ma_fast', 'ma_slow']
    df['target'] = df['signal'].shift(-1)
    df.dropna(inplace=True)
    X = df[features]
    y = df['target']
    X_scaled = SCALER.fit_transform(X)
    model = RandomForestClassifier(n_estimators=50, random_state=1)
    model.fit(X_scaled, y)
    return model

def calculate_min_order_size_usd(current_price):
    """Calculate minimum order size in USD based on BTC price and Kraken's minimum"""
    return MIN_ORDER_SIZE_BTC * current_price

# Track the bot's state
bot_state = {
    'holding_btc': False,  # Whether we currently hold BTC
    'last_buy_price': 0,   # Price at which we last bought BTC
    'buy_amount': 0,       # Amount of BTC we bought
    'trades_executed': 0,  # Number of trades executed
    'total_profit': 0      # Total profit made
}

def analyze_24h_market(latest_df):
    """Analyze the previous 24 hours of market data to determine if current price is a good buying opportunity"""
    # For 1-minute candles, we need 1440 candles to cover 24 hours
    # For other timeframes, adjust accordingly
    required_candles = 1440 if TIMEFRAME == '1m' else 24 if TIMEFRAME == '1h' else 96 if TIMEFRAME == '15m' else 288 if TIMEFRAME == '5m' else 144

    if len(latest_df) < required_candles:
        print(f"Warning: Not enough data for full 24-hour analysis. Using available {len(latest_df)} candles.")
        lookback = len(latest_df)
    else:
        lookback = required_candles

    # Get the last 24 hours of price data
    day_prices = latest_df['Close'].tail(lookback).values
    current_price = day_prices[-1]

    # Calculate key statistics
    day_high = max(day_prices)
    day_low = min(day_prices)
    day_avg = sum(day_prices) / len(day_prices)
    day_range = day_high - day_low

    # Calculate where current price is in the 24h range (0 = at low, 1 = at high)
    if day_range > 0:
        price_position = (current_price - day_low) / day_range
    else:
        price_position = 0.5  # If no range, assume middle position

    # Calculate volatility (standard deviation as percentage of average price)
    volatility = np.std(day_prices) / day_avg if day_avg > 0 else 0

    # Determine if current price is unusually low
    # Consider it unusually low if in the bottom 20% of the 24h range
    is_unusually_low = price_position < 0.2

    # Calculate how far below average (as a percentage)
    below_avg_pct = (day_avg - current_price) / day_avg if current_price < day_avg else 0

    # Prepare detailed analysis
    analysis = {
        'current_price': current_price,
        'day_high': day_high,
        'day_low': day_low,
        'day_avg': day_avg,
        'price_position': price_position,
        'volatility': volatility,
        'below_avg_pct': below_avg_pct,
        'is_unusually_low': is_unusually_low
    }

    return is_unusually_low, analysis

def check_for_dip(latest_df, lookback=5):
    """Check if there's a price dip that's good for buying"""
    if len(latest_df) < lookback:
        return False, 0

    # Calculate short-term price movement
    recent_prices = latest_df['Close'].tail(lookback).values
    price_change = (recent_prices[-1] / recent_prices[0]) - 1

    # Consider it a dip if price has decreased recently
    is_dip = price_change < -0.001  # 0.1% decrease

    return is_dip, price_change

def check_for_profit_opportunity(current_price):
    """Check if there's a profit opportunity based on our last buy price"""
    if not bot_state['holding_btc'] or bot_state['last_buy_price'] == 0:
        return False, 0

    # Calculate potential profit
    # We need to account for fees on both buy and sell transactions
    buy_fee = bot_state['last_buy_price'] * bot_state['buy_amount'] * TRADING_FEE
    sell_fee = current_price * bot_state['buy_amount'] * TRADING_FEE
    total_fees = buy_fee + sell_fee

    # Calculate gross profit (before fees)
    gross_profit = (current_price - bot_state['last_buy_price']) * bot_state['buy_amount']

    # Calculate net profit (after fees)
    net_profit = gross_profit - total_fees

    # Calculate profit percentage
    invested_amount = bot_state['last_buy_price'] * bot_state['buy_amount']
    profit_percentage = net_profit / invested_amount if invested_amount > 0 else 0

    # Only return true if we have a guaranteed positive profit after all fees
    return net_profit > 0 and profit_percentage > PROFIT_THRESHOLD, profit_percentage

def execute_buy(current_price, amount_buy, order_size_usd, simulation_mode=False):
    """Execute a buy order"""
    if simulation_mode:
        print(f"SIMULATION MODE: Would buy {amount_buy:.6f} BTC @ ${current_price}")
        print(f"Total cost: ${order_size_usd:.2f}")

        # Update bot state in simulation mode
        bot_state['holding_btc'] = True
        bot_state['last_buy_price'] = current_price
        bot_state['buy_amount'] = amount_buy
        bot_state['trades_executed'] += 1

        return True
    else:
        try:
            # Check balance before trading
            balance = exchange.fetch_balance()
            usd_balance = balance['USD']['free'] if 'USD' in balance else balance['ZUSD']['free'] if 'ZUSD' in balance else 0

            # Check open orders count
            open_orders = exchange.fetch_open_orders(SYMBOL)
            if len(open_orders) >= MAX_OPEN_ORDERS:
                print(f"⚠️ Maximum open orders ({MAX_OPEN_ORDERS}) reached. Cannot place new order.")
                return False

            if usd_balance >= order_size_usd:
                # Execute the trade
                print(f"Placing order to buy {amount_buy:.6f} BTC (${order_size_usd:.2f})...")
                order = exchange.create_market_buy_order(SYMBOL, amount_buy)
                print(f"✅ REAL TRADE: Bought {amount_buy:.6f} BTC @ ${current_price}")
                print(f"Order ID: {order['id']}")

                # Update bot state
                bot_state['holding_btc'] = True
                bot_state['last_buy_price'] = current_price
                bot_state['buy_amount'] = amount_buy
                bot_state['trades_executed'] += 1

                return True
            else:
                print(f"⚠️ Insufficient balance: ${usd_balance} available, ${order_size_usd:.2f} needed")
                return False
        except Exception as e:
            print(f"❌ Trading error: {e}")
            return False

def execute_sell(current_price, amount_sell, simulation_mode=False):
    """Execute a sell order"""
    # Calculate fees
    buy_fee = bot_state['last_buy_price'] * amount_sell * TRADING_FEE
    sell_fee = current_price * amount_sell * TRADING_FEE
    total_fees = buy_fee + sell_fee

    # Calculate gross profit (before fees)
    gross_profit = (current_price - bot_state['last_buy_price']) * amount_sell

    # Calculate net profit (after fees)
    net_profit = gross_profit - total_fees

    # Only proceed if we have a guaranteed positive profit
    if net_profit <= 0:
        print(f"⚠️ SELL PREVENTED: Would result in a loss of ${-net_profit:.2f}")
        return False

    if simulation_mode:
        print(f"SIMULATION MODE: Would sell {amount_sell:.6f} BTC @ ${current_price}")
        print(f"Buy price: ${bot_state['last_buy_price']:.2f} | Sell price: ${current_price:.2f}")
        print(f"Gross profit: ${gross_profit:.2f} | Fees: ${total_fees:.2f}")
        print(f"Net profit: ${net_profit:.2f} (guaranteed positive)")

        # Update bot state in simulation mode
        bot_state['holding_btc'] = False
        bot_state['total_profit'] += net_profit
        bot_state['trades_executed'] += 1

        return True
    else:
        try:
            # Check BTC balance
            balance = exchange.fetch_balance()
            btc_balance = balance['BTC']['free'] if 'BTC' in balance else 0

            if btc_balance >= amount_sell:
                # Execute the trade
                print(f"Placing order to sell {amount_sell:.6f} BTC @ ${current_price}...")
                order = exchange.create_market_sell_order(SYMBOL, amount_sell)
                print(f"✅ REAL TRADE: Sold {amount_sell:.6f} BTC @ ${current_price}")
                print(f"Order ID: {order['id']}")

                # Display profit details
                print(f"Buy price: ${bot_state['last_buy_price']:.2f} | Sell price: ${current_price:.2f}")
                print(f"Gross profit: ${gross_profit:.2f} | Fees: ${total_fees:.2f}")
                print(f"Net profit: ${net_profit:.2f} (guaranteed positive)")

                # Update bot state
                bot_state['holding_btc'] = False
                bot_state['total_profit'] += net_profit
                bot_state['trades_executed'] += 1

                return True
            else:
                print(f"⚠️ Insufficient BTC balance: {btc_balance} available, {amount_sell:.6f} needed")
                return False
        except Exception as e:
            print(f"❌ Trading error: {e}")
            return False

def trade(model, latest_df, simulation_mode=False, force_buy=False):
    """Main trading function implementing the buy-dip-sell-profit strategy"""
    # Get current price
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        print(f"Current BTC price: ${current_price}")

        # Use the minimum BTC amount allowed by Kraken
        amount_buy = MIN_ORDER_SIZE_BTC
        order_size_usd = amount_buy * current_price

        # Display bot state
        state = "HOLDING BTC" if bot_state['holding_btc'] else "WAITING TO BUY"
        print(f"Bot state: {state} | Trades: {bot_state['trades_executed']} | Profit: ${bot_state['total_profit']:.2f}")

        # Strategy logic
        if bot_state['holding_btc']:
            # We're holding BTC, check if we should sell for profit
            profit_opportunity, profit_percentage = check_for_profit_opportunity(current_price)

            if profit_opportunity:
                print(f"PROFIT OPPORTUNITY DETECTED: {profit_percentage:.2%} (threshold: {PROFIT_THRESHOLD:.2%})")
                print(f"Selling {bot_state['buy_amount']:.6f} BTC bought at ${bot_state['last_buy_price']:.2f}")

                # Execute sell - this will only proceed if profit is guaranteed
                execute_sell(current_price, bot_state['buy_amount'], simulation_mode)
            else:
                # Calculate current profit/loss for display
                buy_fee = bot_state['last_buy_price'] * bot_state['buy_amount'] * TRADING_FEE
                sell_fee = current_price * bot_state['buy_amount'] * TRADING_FEE
                total_fees = buy_fee + sell_fee
                gross_profit = (current_price - bot_state['last_buy_price']) * bot_state['buy_amount']
                net_profit = gross_profit - total_fees

                status = "PROFIT" if net_profit > 0 else "LOSS"
                print(f"Holding BTC. Current {status}: ${net_profit:.2f} ({profit_percentage:.2%})")
                print(f"Need at least {PROFIT_THRESHOLD:.2%} profit to sell")
        else:
            # We're not holding BTC, check if we should buy
            if force_buy:
                print("Executing immediate buy as requested...")
                print("This will start the buy-dip-sell-profit cycle")
                execute_buy(current_price, amount_buy, order_size_usd, simulation_mode)
            else:
                # Check for price dip
                is_dip, price_change = check_for_dip(latest_df)

                # Also use ML model for additional signal
                X_latest = latest_df[['return','ma_fast','ma_slow']]
                X_scaled_latest = SCALER.transform(X_latest)
                prediction = model.predict(X_scaled_latest)[-1]

                # Buy if we detect a dip or ML model predicts a buy
                should_buy = is_dip or prediction == 1

                if should_buy:
                    reason = "price dip" if is_dip else "ML prediction"
                    print(f"BUY SIGNAL: {reason} detected")
                    print(f"Recent price change: {price_change:.2%}")
                    print(f"Will only sell when profit is GUARANTEED after fees")

                    # Execute buy
                    execute_buy(current_price, amount_buy, order_size_usd, simulation_mode)
                else:
                    print("🚫 No buy signal. Waiting for a dip or positive ML prediction.")
                    print(f"Total profit so far: ${bot_state['total_profit']:.2f} from {bot_state['trades_executed']} trades")

    except Exception as e:
        print(f"Error in trading function: {e}")

if __name__ == "__main__":
    import argparse
    import sys
    import datetime

    print("Python version:", sys.version)
    print("CCXT version:", ccxt.__version__)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Kraken Crypto Trading Bot')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default: simulation mode)')
    parser.add_argument('--interval', type=float, default=SCAN_INTERVAL,
                        help=f'Scanning interval in seconds (default: {SCAN_INTERVAL})')
    parser.add_argument('--profit', type=float, default=PROFIT_THRESHOLD,
                        help=f'Profit threshold after fees (default: {PROFIT_THRESHOLD})')
    parser.add_argument('--buy-now', action='store_true',
                        help='Buy immediately when starting the bot')
    args = parser.parse_args()

    # Get command line parameters
    simulation_mode = not args.live
    scan_interval = args.interval
    profit_threshold = args.profit
    buy_immediately = args.buy_now

    # Update the global profit threshold
    PROFIT_THRESHOLD = profit_threshold  # No need for global declaration here

    if simulation_mode:
        print("Starting Kraken Crypto Trading Bot in SIMULATION MODE")
        print("==================================================")
        print("This bot will make trading recommendations but will NOT execute real trades.")
        print("To enable real trading, run with the --live flag.")
    else:
        print("Starting Kraken Crypto Trading Bot in LIVE TRADING MODE")
        print("==================================================")
        print("WARNING: This bot will execute REAL trades with REAL money.")
        print("Press Ctrl+C to stop the bot at any time.")

    print(f"Scanning market every {scan_interval} seconds ({1/scan_interval:.1f} times per second)")
    print("==================================================")

    try:
        # Initial data fetch and model training
        print("Fetching initial data and training model...")
        df = fetch_historical_data()
        df = generate_features(df)
        model = train_ml_model(df)
        print("Model trained successfully.")

        # Track trades and performance
        trades_executed = 0
        last_model_update = datetime.datetime.now()
        last_full_update = datetime.datetime.now()

        print("\nStarting high-frequency trading...\n")

        # Display trading parameters
        print(f"Trading parameters:")
        print(f"- Minimum BTC order size: {MIN_ORDER_SIZE_BTC} BTC")
        print(f"- Trading fee: {TRADING_FEE:.2%}")
        print(f"- Profit threshold: {PROFIT_THRESHOLD:.2%}")
        print(f"- Maximum open orders: {MAX_OPEN_ORDERS}")

        # Perform 24-hour market analysis
        print("\nAnalyzing 24-hour market data...")
        latest_df = fetch_historical_data()
        latest_df = generate_features(latest_df)

        is_unusually_low, market_analysis = analyze_24h_market(latest_df)

        # Display 24-hour market analysis
        print("\n24-HOUR MARKET ANALYSIS:")
        print(f"Current price: ${market_analysis['current_price']:.2f}")
        print(f"24h High: ${market_analysis['day_high']:.2f}")
        print(f"24h Low: ${market_analysis['day_low']:.2f}")
        print(f"24h Average: ${market_analysis['day_avg']:.2f}")
        print(f"Price position in range: {market_analysis['price_position']*100:.1f}% from bottom")
        print(f"24h Volatility: {market_analysis['volatility']*100:.2f}%")

        if market_analysis['below_avg_pct'] > 0:
            print(f"Price is {market_analysis['below_avg_pct']*100:.2f}% below 24h average")
        else:
            print(f"Price is {-market_analysis['below_avg_pct']*100:.2f}% above 24h average")

        # Decide whether to buy immediately
        should_buy_now = buy_immediately or is_unusually_low

        if should_buy_now:
            reason = "user requested immediate buy" if buy_immediately else "price is unusually low in 24h range"
            print(f"\nExecuting immediate buy because {reason}...")
            trade(model, latest_df, simulation_mode, force_buy=True)
        else:
            print("\nCurrent price is not unusually low. Will wait for a better buying opportunity.")

        # Main trading loop
        while True:
            current_time = datetime.datetime.now()

            # Fetch latest data
            try:
                latest_df = fetch_historical_data(verbose=False)  # Reduce output noise
                latest_df = generate_features(latest_df)

                # Execute trade logic
                trade(model, latest_df, simulation_mode)

                # Retrain model every hour to adapt to changing market conditions
                if (current_time - last_model_update).total_seconds() > 3600:  # 1 hour
                    print("\nRetraining model with fresh data...")
                    model = train_ml_model(latest_df)
                    last_model_update = current_time
                    print("Model updated successfully.")

                # Print full status update every 5 minutes
                if (current_time - last_full_update).total_seconds() > 300:  # 5 minutes
                    print("\n==================================================")
                    print(f"Status update at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Trading {SYMBOL} | {'SIMULATION' if simulation_mode else 'LIVE'} mode")
                    print(f"Bot state: {'HOLDING BTC' if bot_state['holding_btc'] else 'WAITING TO BUY'}")
                    print(f"Trades executed: {bot_state['trades_executed']}")
                    print(f"Total profit: ${bot_state['total_profit']:.2f}")
                    print(f"Scanning interval: {scan_interval} seconds")
                    print("==================================================")
                    last_full_update = current_time

            except Exception as e:
                print(f"Error during trading cycle: {e}")

            # Sleep for the specified interval
            time.sleep(scan_interval)

    except KeyboardInterrupt:
        print("\nBot stopped by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
import ccxt
import time

print("Testing Kraken API connection...")

# Initialize Kraken exchange
kraken = ccxt.kraken()

# Test public API endpoints
print("\nFetching available symbols...")
markets = kraken.load_markets()
print(f"Found {len(markets)} markets")
print(f"BTC/USD available: {'BTC/USD' in markets}")

print("\nFetching ticker for BTC/USD...")
ticker = kraken.fetch_ticker('BTC/USD')
print(f"Current BTC price: ${ticker['last']}")

print("\nFetching OHLCV data for BTC/USD...")
try:
    candles = kraken.fetch_ohlcv('BTC/USD', timeframe='1h', limit=10)
    print(f"Received {len(candles)} candles")
    print(f"First candle: {candles[0]}")
except Exception as e:
    print(f"Error fetching OHLCV data: {e}")

print("\nTest completed successfully!")

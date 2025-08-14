import os
import json
import asyncio
import redis.asyncio as aioredis
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import time

# Configuration
MEMURAI_HOST = "localhost"
MEMURAI_PORT = 6379
DATA_DIR = "realtime_data"
SYMBOL = "BTC-USDT"
MAX_LIST_LENGTH = 50  # Keep recent data only
TIMESTAMP_THRESHOLD = 48 * 3600 * 1000  # 48 hours for testing

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return (macd - signal).fillna(0)

def calculate_obv(df):
    """Calculate On-Balance Volume (OBV)"""
    obv = (np.sign(df['price'].diff()) * df['volume']).cumsum()
    return obv.fillna(0)

def calculate_features(df):
    """Calculate and normalize features"""
    df = df.copy()
    if 'price' not in df.columns:
        raise ValueError("Column 'price' not found in data")
    if 'amount' not in df.columns:
        print("⚠️ Column 'amount' not found. Setting default volume to 1.")
        df['volume'] = 1
    else:
        df['volume'] = df['amount']

    df['log_return'] = np.log(df['price']).diff().fillna(0)
    df['sma'] = df['price'].rolling(window=3).mean().fillna(df['price'].iloc[0])
    df['ema'] = df['price'].ewm(span=3, adjust=False).mean()
    df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['volatility'] = df['log_return'].rolling(window=5).std().fillna(0)
    df['rsi'] = calculate_rsi(df['price'], window=14)
    df['macd'] = calculate_macd(df['price'])
    df['obv'] = calculate_obv(df)
    
    scaler = RobustScaler()
    features = ['price', 'log_return', 'sma', 'ema', 'vwap', 'volatility', 'rsi', 'macd', 'obv']
    for col in features:
        if col in df and df[col].std() != 0 and not df[col].isnull().all():
            df[col] = scaler.fit_transform(df[[col]]).flatten()
        else:
            df[col] = df[col].fillna(0)
    return df

async def update_memurai():
    """Update Memurai with processed features"""
    redis = aioredis.Redis(host=MEMURAI_HOST, port=MEMURAI_PORT, decode_responses=True)
    memurai_key = f"features:{SYMBOL}"

    processed_files = set()
    last_timestamps = set()

    # Clear Memurai at start
    await redis.delete(memurai_key)

    while True:
        all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json') and SYMBOL in f])
        new_files = [f for f in all_files if f not in processed_files]

        if not new_files:
            print("⏳ No new files found for processing. Waiting...")
        else:
            current_time = int(time.time() * 1000)
            for file_name in new_files:
                path = os.path.join(DATA_DIR, file_name)
                print(f"⏳ Processing new file: {file_name}")

                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    print(f"JSON content of {file_name}:")
                    print(json.dumps(data[:5], indent=2, ensure_ascii=False))

                    df = pd.DataFrame(data)
                    print(f"DataFrame columns for {file_name}: {df.columns.tolist()}")
                    
                    # Filter old data
                    df = df[df['timestamp'] >= current_time - TIMESTAMP_THRESHOLD]
                    if df.empty:
                        print(f"⚠️ File {file_name} contains only old data. Skipping.")
                        processed_files.add(file_name)
                        continue

                    # Check for duplicate timestamps
                    current_timestamps = set(df['timestamp'])
                    if current_timestamps.issubset(last_timestamps):
                        print(f"⚠️ File {file_name} contains duplicate data. Skipping.")
                        processed_files.add(file_name)
                        continue
                    last_timestamps.update(current_timestamps)

                    df = calculate_features(df)
                    print(f"Last 5 feature rows for {file_name}:")
                    print(df.tail(5))
                    records = df.to_dict(orient='records')

                    pipe = redis.pipeline()
                    for record in records:
                        pipe.rpush(memurai_key, json.dumps(record))
                    pipe.ltrim(memurai_key, -MAX_LIST_LENGTH, -1)
                    await pipe.execute()

                    print(f"✅ File {file_name} successfully added to Memurai.")
                    processed_files.add(file_name)

                except Exception as e:
                    print(f"❌ Error processing or writing file {file_name}: {e}")

        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(update_memurai())
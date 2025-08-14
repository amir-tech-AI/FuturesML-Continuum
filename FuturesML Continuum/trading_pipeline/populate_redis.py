import pandas as pd
import redis.asyncio as redis
import asyncio
import json

async def populate_redis_from_csv(csv_path, symbol='BTCUSDT'):
    df = pd.read_csv(csv_path)
    features = df[['price', 'log_return', 'sma', 'vwap', 'volatility']].to_dict('records')
    features_dict = {str(i): row for i, row in enumerate(features)}
    
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    await redis_client.set(f"features:{symbol}", json.dumps(features_dict))
    await redis_client.aclose()
    print(f"Features stored in Redis for {symbol}")

if __name__ == "__main__":
    asyncio.run(populate_redis_from_csv('historical_data_with_features.csv'))
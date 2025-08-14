import asyncio
import json
import aioredis
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class RealtimeFeatureReader:
    def __init__(self, redis_url="redis://localhost"):
        self.redis_url = redis_url
        self.redis = None

    async def connect(self):
        self.redis = await aioredis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        logging.info(f"Connected to Redis at {self.redis_url}")

    async def get_latest_features(self, symbol: str):
        """
        خواندن جدیدترین فیچرها و تیک از Redis به صورت دیکشنری Python
        """
        key = f"futuresml:{symbol.upper()}:latest"
        data_json = await self.redis.get(key)
        if not data_json:
            logging.warning(f"No data found in Redis for key: {key}")
            return None
        try:
            data = json.loads(data_json)
            return data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from Redis for key {key}: {e}")
            return None

    async def close(self):
        if self.redis:
            await self.redis.close()
            logging.info("Redis connection closed")

async def main():
    reader = RealtimeFeatureReader(redis_url="redis://localhost")
    await reader.connect()

    symbol = input("Enter symbol to fetch features (e.g. BTC-USDT): ").strip().upper()

    # نمونه خواندن و چاپ فیچر
    data = await reader.get_latest_features(symbol)
    if data:
        print(f"Latest features for {symbol}:")
        print(json.dumps(data['features'], indent=2))
        print(f"Last tick:")
        print(json.dumps(data['last_tick'], indent=2))
        print(f"Timestamp: {data.get('timestamp')}")
    else:
        print("No data available.")

    await reader.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Program stopped by user.")

import asyncio
import aioredis
import json

MEMURAI_HOST = "localhost"
MEMURAI_PORT = 6379
REDIS_KEY = "features:BTCUSDT"

async def check_data():
    redis = await aioredis.from_url(f"redis://{MEMURAI_HOST}:{MEMURAI_PORT}", decode_responses=True)
    raw_data = await redis.get(REDIS_KEY)
    if raw_data is None:
        print("No data in Memurai!")
        return
    data = json.loads(raw_data)
    print(f"Number of entries: {len(data)}")
    print("Sample entry:", data[-1])

asyncio.run(check_data())

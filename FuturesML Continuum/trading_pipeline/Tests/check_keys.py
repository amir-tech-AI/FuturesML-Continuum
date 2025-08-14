import asyncio
import aioredis

async def get_redis_connection():
    return await aioredis.from_url("redis://localhost:6379", decode_responses=True)

async def list_keys():
    redis = await get_redis_connection()
    keys = await redis.keys("*")
    await redis.close()
    return keys

if __name__ == "__main__":
    keys = asyncio.run(list_keys())
    print("Keys in Memurai:", keys)

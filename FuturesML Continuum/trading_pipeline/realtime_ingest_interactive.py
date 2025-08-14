import asyncio
import json
import gzip
import websockets
import logging
from collections import deque
from typing import Deque, Dict, Any, List
import numpy as np
import datetime
import aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class RealtimeBuffer:
    def __init__(self, max_size: int = 1000):
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=max_size)
    
    def add_tick(self, tick: Dict[str, Any]):
        self.buffer.append(tick)
        logging.debug(f"Tick added. Buffer size: {len(self.buffer)}")
    
    def get_all(self) -> List[Dict[str, Any]]:
        return list(self.buffer)

class FeatureEngineer:
    def __init__(self, window_sma: int = 5, window_vol: int = 10):
        self.window_sma = window_sma
        self.window_vol = window_vol
    
    def compute_features(self, ticks: List[Dict[str, Any]]) -> Dict[str, float]:
        if len(ticks) < 2:
            return {}

        prices = np.array([tick['price'] for tick in ticks])
        volumes = np.array([tick['amount'] for tick in ticks])

        log_return = np.log(prices[-1]) - np.log(prices[-2])
        sma = np.mean(prices[-self.window_sma:]) if len(prices) >= self.window_sma else np.mean(prices)

        vwap = np.sum(prices[-self.window_sma:] * volumes[-self.window_sma:]) / np.sum(volumes[-self.window_sma:]) if np.sum(volumes[-self.window_sma:]) > 0 else float('nan')

        if len(prices) >= self.window_vol + 1:
            log_returns = np.diff(np.log(prices[-(self.window_vol+1):]))
            volatility = np.std(log_returns)
        else:
            volatility = float('nan')

        return {
            'log_return': float(log_return),
            'sma': float(sma),
            'vwap': float(vwap),
            'volatility': float(volatility)
        }

class MultiSymbolRealtimeIngestRedis:
    def __init__(self, symbols: List[str], buffer_size: int = 1000, save_interval_sec: int = 10, redis_url: str = "redis://localhost"):
        self.symbols = [sym.upper() for sym in symbols]
        self.buffers = {symbol: RealtimeBuffer(max_size=buffer_size) for symbol in self.symbols}
        self.feature_engineers = {symbol: FeatureEngineer(window_sma=5, window_vol=10) for symbol in self.symbols}
        self.save_interval = save_interval_sec
        self.redis_url = redis_url
        self.redis = None
        self.ws_url = "wss://api.hbdm.com/linear-swap-ws"  # جهت USDT-Margined، در صورت نیاز تغییر دهید

    async def connect_redis(self):
        self.redis = await aioredis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        logging.info(f"Connected to Redis at {self.redis_url}")

    async def save_features_to_redis(self, symbol: str, features: Dict[str, float], last_tick: Dict[str, Any]):
        """
        ذخیره داده های آخرین فیچر و تیک به عنوان JSON در Redis با کلید مجزا
        """
        try:
            data = {
                'features': features,
                'last_tick': last_tick,
                'timestamp': datetime.datetime.utcnow().isoformat()
            }
            key = f"futuresml:{symbol}:latest"
            await self.redis.set(key, json.dumps(data))
            logging.info(f"[{symbol}] Saved latest features to Redis key: {key}")
        except Exception as e:
            logging.error(f"Error saving features to Redis for {symbol}: {e}")

    async def periodic_redis_connection_check(self):
        while True:
            try:
                pong = await self.redis.ping()
                logging.debug(f"Redis ping: {pong}")
            except Exception as e:
                logging.error(f"Redis connection lost: {e}. Reconnecting...")
                await self.connect_redis()
            await asyncio.sleep(30)  # هر 30 ثانیه چک میکند

    async def connect_and_receive(self):
        await self.connect_redis()
        reconnect_delay = 5
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20) as ws:
                    logging.info(f"Connected to WebSocket {self.ws_url}")
                    for symbol in self.symbols:
                        topic = f"market.{symbol}.trade.detail"
                        sub_msg = json.dumps({"sub": topic, "id": f"subscribe-{symbol}"})
                        await ws.send(sub_msg)
                        logging.info(f"Subscribe sent: {topic}")

                    while True:
                        msg = await ws.recv()
                        if isinstance(msg, bytes):
                            msg = gzip.decompress(msg).decode('utf-8')
                        data = json.loads(msg)

                        if "ping" in data:
                            pong_msg = json.dumps({"pong": data["ping"]})
                            await ws.send(pong_msg)
                            continue

                        if "status" in data and data["status"] == "ok":
                            logging.info(f"Subscription confirmed: {data.get('subbed')}")
                            continue

                        ch = data.get("ch", "")
                        if "tick" in data and "data" in data.get("tick", {}):
                            parts = ch.split('.')
                            if len(parts) >= 2:
                                symbol = parts[1].upper()
                                if symbol not in self.symbols:
                                    logging.warning(f"Received data for unsubscribed symbol {symbol}")
                                else:
                                    trades = data["tick"]["data"]
                                    buffer = self.buffers[symbol]
                                    feat_eng = self.feature_engineers[symbol]
                                    
                                    last_tick = None
                                    for trade in trades:
                                        tick = {
                                            "trade_id": trade.get("tradeId"),
                                            "price": trade.get("price"),
                                            "amount": trade.get("amount"),
                                            "direction": trade.get("direction"),
                                            "timestamp": trade.get("ts")
                                        }
                                        buffer.add_tick(tick)
                                        last_tick = tick
                                    
                                    ticks = buffer.get_all()
                                    features = feat_eng.compute_features(ticks)
                                    
                                    logging.info(f"[{symbol}] Latest tick count: {len(buffer.buffer)} | Features: {features}")

                                    # ذخیره فیچر و تیک آخرین در Redis
                                    if self.redis and features and last_tick:
                                        await self.save_features_to_redis(symbol, features, last_tick)

                        elif "status" in data and data["status"] == "error":
                            logging.error(f"Error from server: {data.get('err-msg')}")
                        else:
                            logging.debug(f"Other message received: {data}")

            except (websockets.exceptions.ConnectionClosed, ConnectionResetError) as e:
                logging.warning(f"Connection lost: {e} — reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
            except Exception as e:
                logging.error(f"Unexpected error: {e} — reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)

    async def run(self):
        # اجرا به صورت همزمان اتصال WebSocket و بررسی اتصال Redis
        await asyncio.gather(
            self.connect_and_receive(),
            self.periodic_redis_connection_check()
        )

async def main():
    print("FuturesML Continuum Multi-symbol Realtime Ingest with Redis")
    symbols_input = input("Enter symbols to subscribe (comma separated, e.g. BTC-USDT,ETH-USDT): ")
    symbols = [sym.strip().upper() for sym in symbols_input.split(",") if sym.strip()]

    if not symbols:
        print("No valid symbols entered. Exiting.")
        return

    ingest = MultiSymbolRealtimeIngestRedis(symbols=symbols)
    await ingest.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Realtime ingest stopped by user.")

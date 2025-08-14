import asyncio
import json
import aioredis
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Configuration
MEMURAI_HOST = "localhost"
MEMURAI_PORT = 6379
SYMBOLS = ["BTC-USDT"]

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

async def get_features_from_redis(symbol: str):
    """Retrieve features from Memurai/Redis for the given symbol."""
    redis = await aioredis.from_url(f"redis://{MEMURAI_HOST}:{MEMURAI_PORT}", decode_responses=True)
    key = f"features:{symbol}"
    raw_list = await redis.lrange(key, -50, -1)  # Read last 50 records

    if not raw_list:
        print(f"⚠️ No data available in Memurai for {symbol}.")
        return None

    data = [json.loads(item) for item in raw_list]
    print(f"📊 Retrieved {len(data)} records for {symbol}.")
    
    # Print last 5 samples for debugging
    print(f"Last 5 samples for {symbol}:")
    for sample in data[-5:]:
        print(sample)

    feature_columns = ['price', 'log_return', 'sma', 'vwap', 'volatility']
    features = []
    for f in data:
        feature_values = [f.get(col, 0) for col in feature_columns]
        if not all(isinstance(v, (int, float)) for v in feature_values):
            print(f"⚠️ Invalid feature values in sample: {f}")
            continue
        features.append(feature_values)

    if not features:
        print(f"⚠️ No valid feature data for {symbol}.")
        return None

    return features

async def main_loop():
    """Main loop for real-time predictions."""
    device = torch.device("cpu")  # Change to "cuda" if GPU is available

    model = LSTMModel(input_size=5, hidden_size=64, num_layers=2, output_size=1)
    try:
        model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    model.to(device)
    model.eval()

    while True:
        for symbol in SYMBOLS:
            features = await get_features_from_redis(symbol)
            if features is None or len(features) == 0:
                print(f"⚠️ No data available for {symbol}.")
                await asyncio.sleep(5)
                continue

            if len(features) < 50:
                print(f"⚠️ Not enough data points for {symbol} ({len(features)}). Need 50.")
                await asyncio.sleep(5)
                continue

            print(f"Input features (last sample): {features[-1]}")
            print(f"Number of input samples: {len(features)}")

            input_tensor = torch.tensor(np.array(features), dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(input_tensor).item()
                print(f"📈 Predicted value for {symbol}: {prediction:.6f}")

        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main_loop())
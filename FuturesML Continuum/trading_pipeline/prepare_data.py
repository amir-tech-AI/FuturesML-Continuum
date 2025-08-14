import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json

def prepare_historical_data(csv_file, sequence_length=30, batch_size=32):
    df = pd.read_csv(csv_file)

    # فقط ستون‌های فیچر
    features = df[['price', 'log_return', 'sma', 'vwap', 'volatility']].values

    # محاسبه mean و std کل دیتاست
    mean_vals = features.mean(axis=0)
    std_vals = features.std(axis=0)
    std_vals[std_vals == 0] = 1e-8  # جلوگیری از تقسیم بر صفر

    # ذخیره برای استفاده در پیش‌بینی
    scaler_stats = {
        "mean": mean_vals.tolist(),
        "std": std_vals.tolist()
    }
    with open("scaler_stats.json", "w") as f:
        json.dump(scaler_stats, f)

    # نرمال‌سازی کل دیتاست
    norm_features = (features - mean_vals) / std_vals

    # ساخت داده‌های sequence
    sequences = []
    labels = []
    for i in range(len(norm_features) - sequence_length):
        seq = norm_features[i:i+sequence_length]
        label = 1 if df['price'].iloc[i+sequence_length] > df['price'].iloc[i+sequence_length-1] else 0
        sequences.append(seq)
        labels.append(label)

    sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.float32)

    dataset = TensorDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, (mean_vals, std_vals)

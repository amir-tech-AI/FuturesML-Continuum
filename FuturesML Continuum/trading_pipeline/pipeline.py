import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path


def download_market_data(symbol: str, start_date: str, end_date: str, interval='1d') -> pd.DataFrame:
    """
    دانلود داده‌های تاریخی بازار از yfinance
    """
    print(f"Downloading market data for {symbol} from {start_date} to {end_date} ...")
    data = yf.download(tickers=symbol, start=start_date, end=end_date, interval=interval)
    data.reset_index(inplace=True)
    print(f"Data downloaded: {len(data)} rows.")
    return data


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    محاسبه فیچرهای تکنیکال پایه روی داده بازار
    """
    print("Computing features ...")
    df = df.sort_values('Date').reset_index(drop=True)

    # میانگین متحرک ساده (SMA)
    df['sma_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['sma_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['sma_diff'] = df['sma_5'] - df['sma_20']

    # بازده لگاریتمی
    df['log_return'] = np.log(df['Close']).diff().fillna(0)

    # VWAP ساده (وزن‌دار بر حسب حجم معاملات)
    df['vwap'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    print("Features computed.")
    return df


def save_parquet(df: pd.DataFrame, filepath: str):
    """
    ذخیره داده و فیچرها به صورت فایل Parquet
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, compression='snappy')
    print(f"Data saved to {filepath}")


if __name__ == "__main__":
    # تنظیمات اصلی پروژه
    symbol = "BTC-USD"                # نماد دارایی روی Yahoo Finance
    start_date = "2023-01-01"        # تاریخ شروع
    end_date = "2023-08-10"          # تاریخ پایان (می‌توانید تغییر دهید)
    interval = "1d"                  # فواصل داده (روزانه)

    # دانلود داده‌های بازار
    market_data = download_market_data(symbol, start_date, end_date, interval)

    # محاسبه فیچرها
    data_with_features = compute_features(market_data)

    # ذخیره سازی در فایل Parquet
    save_path = f"data/{symbol}_features.parquet"
    save_parquet(data_with_features, save_path)

    # نمایش چند سطر آخر برای تایید
    print("\nSample of processed data with features:")
    print(data_with_features.tail())

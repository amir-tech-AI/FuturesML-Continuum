import pandas as pd
import numpy as np

def calculate_features(df):
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df['log_return'].fillna(0, inplace=True)

    # SMA و Volatility با min_periods=1 تا از همان ابتدا محاسبه شوند
    df['sma'] = df['price'].rolling(window=20, min_periods=1).mean()
    df['vwap'] = (df['price'] * df['amount']).cumsum() / df['amount'].cumsum()
    df['volatility'] = df['log_return'].rolling(window=20, min_periods=1).std().fillna(0)

    return df


def add_features_to_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df = calculate_features(df)
    # فقط ستون‌های موردنیاز رو نگه می‌داریم
    df = df[['timestamp', 'price', 'amount', 'log_return', 'sma', 'vwap', 'volatility']]
    df.to_csv(output_csv, index=False)
    print(f"CSV with features saved to {output_csv}")

if __name__ == "__main__":
    add_features_to_csv('historical_data.csv', 'historical_data_with_features.csv')
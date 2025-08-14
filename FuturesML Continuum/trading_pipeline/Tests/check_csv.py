# فایل: check_csv.py
import pandas as pd

df = pd.read_csv('historical_data.csv')
print("ستون‌های فایل CSV:")
print(df.columns)
print("\nچند ردیف اول داده‌ها:")
print(df.head())
import os

DATA_DIR = "realtime_data"
SYMBOL = "BTCUSDT"

files = sorted(os.listdir(DATA_DIR))
print("Files in folder:")
for f in files:
    print(f)

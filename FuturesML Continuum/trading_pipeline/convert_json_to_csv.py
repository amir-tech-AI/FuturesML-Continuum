# فایل: convert_json_to_csv.py
import pandas as pd
import os
import glob

def convert_json_to_csv(json_dir, output_csv):
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    all_data = []
    for file in json_files:
        df = pd.read_json(file)
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv}")

if __name__ == "__main__":
    json_dir = 'realtime_data'
    output_csv = 'historical_data.csv'
    convert_json_to_csv(json_dir, output_csv)
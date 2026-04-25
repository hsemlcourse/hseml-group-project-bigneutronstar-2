import yfinance as yf
import pandas as pd
import argparse
from pathlib import Path
def fetch_gold_data(interval="1h", period="730d", output_path="data/gold_data_1h_cleaned.csv"):
    """
    Downloads historical gold futures data from Yahoo Finance.
    Gold Futures ticker is GC=F.
    """
    print(f"Downloading GC=F (Gold Futures) data... Interval: {interval}, Period: {period}")
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        print("Error: Downloaded data is empty. Check your internet connection or limits.")
        return
    df.reset_index(inplace=True)
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "DateTime"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "DateTime"}, inplace=True)
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    if "Dividends" in df.columns:
        df.drop(columns=["Dividends"], inplace=True)
    if "Stock Splits" in df.columns:
        df.drop(columns=["Stock Splits"], inplace=True)
    print(f"Downloaded {len(df)} rows.")
    print("Saving to", output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Done!")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Gold data from Yahoo Finance.")
    parser.add_argument("--interval", type=str, default="1h", help="Data interval (e.g., 1h, 1d)")
    parser.add_argument("--period", type=str, default="730d", help="Data period (e.g., 730d, max)")
    parser.add_argument("--out", type=str, default="data/gold_data_1h_cleaned.csv", help="Output file path")
    args = parser.parse_args()
    fetch_gold_data(interval=args.interval, period=args.period, output_path=args.out)

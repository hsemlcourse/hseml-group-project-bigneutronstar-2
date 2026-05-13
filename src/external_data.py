import os
import pandas as pd
import yfinance as yf
from datetime import timedelta
import logging

def setup_logger():
    logger = logging.getLogger("external_data")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

logger = setup_logger()

def get_date_range(file_path):
    """
    Get the date range from the specified dataset file.
    Returns (start_date, end_date) localized to UTC.
    """
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if 'DateTime' in df.columns:
                dt_series = pd.to_datetime(df['DateTime'], utc=True)
                min_date = dt_series.min()
                max_date = dt_series.max()
                
                start_date = min_date - timedelta(days=3)
                end_date = max_date + timedelta(days=3)
                
                logger.info(f"Found dataset at {file_path}. Target date range: {start_date} to {end_date}")
                return start_date, end_date
            else:
                logger.warning(f"'DateTime' column not found in {file_path}.")
        except Exception as e:
            logger.warning(f"Could not read or parse {file_path}. Error: {e}")
    else:
        logger.warning(f"File {file_path} does not exist.")
            
    end_date = pd.Timestamp.utcnow()
    start_date = end_date - timedelta(days=730)
    logger.info(f"Using fallback date range: {start_date} to {end_date}")
    return start_date, end_date

def fetch_ticker_data(ticker, start_date, end_date, interval):
    """
    Fetch ticker data from yfinance.
    Normalizes columns and sets timezone to UTC.
    """
    logger.info(f"Fetching {ticker} with interval {interval}...")
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        data = yf.download(ticker, start=start_str, end=end_str, interval=interval, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        if data.empty:
            logger.warning(f"Data for {ticker} is empty for interval {interval}")
            return None
            
        data = data.reset_index()
        
        datetime_col = None
        for col in ['Date', 'Datetime', 'index']:
            if col in data.columns:
                datetime_col = col
                break
                
        if datetime_col is None:
            logger.warning(f"Could not find a valid date/time column for {ticker} at {interval}")
            return None
            
        data.rename(columns={datetime_col: 'DateTime'}, inplace=True)
        
        data['DateTime'] = pd.to_datetime(data['DateTime'], utc=True)
            
        data.sort_values('DateTime', inplace=True)
        
        data = data.loc[:, ~data.columns.duplicated()]
        
        return data
        
    except Exception as e:
        logger.warning(f"Error fetching {ticker} at {interval}: {e}")
        return None

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gold_data_path = os.path.join(base_dir, 'data', 'raw', 'gold_data_1h_cleaned.csv')
    
    start_date, end_date = get_date_range(gold_data_path)
    
    tickers = {
        'dxy': 'DX-Y.NYB',
        'vix': '^VIX',
        'tnx': '^TNX',
        'silver': 'SI=F',
        'oil': 'CL=F'
    }
    
    out_dir = os.path.join(base_dir, 'data', 'raw', 'external')
    os.makedirs(out_dir, exist_ok=True)
    
    downloaded_data = {}
    reports = []
    
    for name, ticker in tickers.items():
        # First try 1h
        data = fetch_ticker_data(ticker, start_date, end_date, "1h")
        interval_used = "1h"
        
        if data is None or len(data) < 10:
            logger.info(f"1h data for {ticker} is insufficient or failed. Falling back to 1d.")
            data = fetch_ticker_data(ticker, start_date, end_date, "1d")
            interval_used = "1d"
            
        if data is not None and not data.empty:
            file_name = f"{name}_{interval_used}.csv"
            out_path = os.path.join(out_dir, file_name)
            data.to_csv(out_path, index=False)
            logger.info(f"Saved {file_name} to {out_dir}")
            
            if 'Close' in data.columns:
                subset = data[['DateTime', 'Close']].copy()
                subset.rename(columns={'Close': name}, inplace=True)
                downloaded_data[name] = subset
                
            n_rows = len(data)
            n_missing_close = data['Close'].isna().sum() if 'Close' in data.columns else 0
            
            reports.append({
                'Name': name,
                'Ticker': ticker,
                'Interval': interval_used,
                'Rows': n_rows,
                'Missing_Close': n_missing_close,
                'Min_Date': data['DateTime'].min(),
                'Max_Date': data['DateTime'].max(),
                'Status': 'Success'
            })
        else:
            logger.warning(f"Failed to fetch any data for {ticker}.")
            reports.append({
                 'Name': name,
                 'Ticker': ticker,
                 'Interval': 'Failed',
                 'Rows': 0,
                 'Missing_Close': 0,
                 'Min_Date': None,
                 'Max_Date': None,
                 'Status': 'Failed/Excluded'
            })
            
    # Combine data into a single file
    combined_shape = (0, 0)
    if downloaded_data:
        logger.info("Combining downloaded features into a single dataset...")
        combined_df = None
        for name, df in downloaded_data.items():
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='DateTime', how='outer')
                
        combined_df.sort_values('DateTime', inplace=True)
        
        combined_path = os.path.join(out_dir, 'external_factors_combined.csv')
        combined_df.to_csv(combined_path, index=False)
        combined_shape = combined_df.shape
        logger.info(f"Saved combined file to {combined_path}")
    else:
        logger.warning("No data downloaded. Combined file will not be created.")
        
    print("\n" + "="*60)
    print("DOWNLOAD REPORT")
    print("="*60)
    for rep in reports:
        print(f"[{rep['Name'].upper()}] - Ticker: {rep['Ticker']}")
        print(f"  Status: {rep['Status']}")
        if rep['Status'] == 'Success':
            print(f"  Interval: {rep['Interval']}")
            print(f"  Rows: {rep['Rows']}")
            print(f"  Missing Values in Close: {rep['Missing_Close']}")
            print(f"  Date Range: {rep['Min_Date']} to {rep['Max_Date']}")
        print("-" * 60)
        
    print(f"Combined DataFrame Shape: {combined_shape}")
    print(f"Output Directory: {out_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

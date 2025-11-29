import requests
import pandas as pd
import time
from datetime import datetime

# ==========================================
# 🆓 FREE DATA SOURCE (Blockchain.com)
# ==========================================
# No API Key required. These endpoints are public.

METRICS_MAP = {
    # metric_name_for_csv : blockchain_com_chart_name
    "Adoption_ActiveAddresses": "n-unique-addresses",
    "Adoption_TxCount": "n-transactions",
    "Distribution_MinerRevenue": "miners-revenue",
    "Valuation_MVRV": "mvrv",
    "Market_HashRate": "hash-rate",
    "Market_Difficulty": "difficulty"
}

def fetch_blockchain_data(metric_name, chart_name):
    """Fetches data from Blockchain.com public charts."""
    url = f"https://api.blockchain.info/charts/{chart_name}"
    params = {
        "timespan": "8years", # Get max history
        "rollingAverage": "24hours",
        "format": "json"
    }
    
    print(f"   Fetching {metric_name}...")
    try:
        # Add a fake user-agent just in case
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, params=params, headers=headers)
        
        if res.status_code == 200:
            data = res.json()
            # Blockchain.com returns values in 'values' list: [{'x': timestamp, 'y': value}]
            df = pd.DataFrame(data['values'])
            
            # 'x' is unix timestamp, 'y' is the value
            df['t'] = pd.to_datetime(df['x'], unit='s')
            df.set_index('t', inplace=True)
            df.rename(columns={'y': metric_name}, inplace=True)
            
            # Drop the 'x' column
            df.drop(columns=['x'], inplace=True)
            return df
        else:
            print(f"   ❌ Error {res.status_code}: {res.text[:100]}")
            return None
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None

def main():
    print("--- 🆓 SAM-LSTM Free Data Fetcher ---")
    print("Fetching real on-chain data from Blockchain.com (No Key Needed)...")
    
    master_df = pd.DataFrame()

    for metric_name, chart_name in METRICS_MAP.items():
        df = fetch_blockchain_data(metric_name, chart_name)
        
        if df is not None:
            if master_df.empty:
                master_df = df
            else:
                # Outer join to align all timestamps
                master_df = master_df.join(df, how='outer')
        
        # Be polite to the free API
        time.sleep(1.0) 

    # Clean up
    if not master_df.empty:
        # Sort by date
        master_df.sort_index(inplace=True)
        
        # Forward fill missing data (charts might have slight time diffs)
        master_df.fillna(method='ffill', inplace=True)
        master_df.dropna(inplace=True) # Drop early rows if data started later
        
        # Save
        filename = f"real_data_free_{datetime.now().strftime('%Y%m%d')}.csv"
        master_df.to_csv(filename)
        
        print(f"\n✅ Success! Data saved to: {filename}")
        print("➡️  Upload this file to the 'Real (CSV)' tab in the Backtester.")
        print(f"   Rows collected: {len(master_df)}")
    else:
        print("\n❌ No data collected. Check internet connection.")

if __name__ == "__main__":
    main()
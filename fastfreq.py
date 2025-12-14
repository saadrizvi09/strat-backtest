import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="HFT Pairs Trader (Real Data)")
st.markdown("""
<style>
    .metric-container {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #333;
        text-align: center;
    }
    .big-num { font-size: 26px; font-weight: bold; color: #00ff41; }
    .label { font-size: 14px; color: #888; }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING (REAL DATA) ---
@st.cache_data(ttl=300) # Cache for 5 mins
def get_real_data(period='7d', interval='1m'):
    """
    Fetches real 1-minute data for BTC and ETH from Yahoo Finance.
    Yahoo limits 1m data to the last 7 days.
    """
    tickers = "BTC-USD ETH-USD"
    data = yf.download(tickers, period=period, interval=interval, group_by='ticker', progress=False)
    
    # Clean and align structure
    df = pd.DataFrame(index=data.index)
    df['BTC'] = data['BTC-USD']['Close']
    df['ETH'] = data['ETH-USD']['Close']
    
    # Drop NaNs (some minutes might be missing)
    df.dropna(inplace=True)
    return df

# --- 2. INDICATOR LOGIC ---
def calculate_zscore(df, window=60):
    """
    Calculates the Z-Score of the ETH/BTC ratio.
    Z-Score measures how many standard deviations the ratio is from its mean.
    """
    # 1. Calculate the Ratio (The Asset)
    df['Ratio'] = df['ETH'] / df['BTC']
    
    # 2. Calculate Rolling Mean & Std Dev
    df['Mean'] = df['Ratio'].rolling(window=window).mean()
    df['Std'] = df['Ratio'].rolling(window=window).std()
    
    # 3. Calculate Z-Score
    # (Current Price - Average Price) / Volatility
    df['Z-Score'] = (df['Ratio'] - df['Mean']) / df['Std']
    
    return df.dropna()

# --- MAIN APP ---

st.title("⚡ Real-Data Frequency Strategy (Pairs Trading)")
st.caption("Strategy: Mean Reversion on ETH/BTC Ratio | Interval: 1 Minute | Source: Yahoo Finance")

# Sidebar
with st.sidebar:
    st.header("⚙️ HFT Settings")
    
    # Tunable Parameters to force trade frequency
    st.info("💡 To get 50+ trades/day, keep the Window LOW and Threshold LOW.")
    
    window = st.slider("Lookback Window (Minutes)", 10, 120, 30)
    threshold = st.slider("Z-Score Trigger", 0.5, 3.0, 1.0, step=0.1)
    
    st.divider()
    
    capital = st.number_input("Capital ($)", value=10000)
    fee = st.number_input("Fee per Trade (%)", value=0.075) / 100
    
    run = st.button("Run Simulation", type="primary")

if run:
    with st.spinner("Fetching 1-minute live data from Yahoo Finance..."):
        raw_df = get_real_data()
        
    if raw_df.empty:
        st.error("Yahoo Finance returned no data. Try again later.")
    else:
        # Process Indicators
        df = calculate_zscore(raw_df.copy(), window)
        
        # --- 3. BACKTEST ENGINE ---
        trades = []
        position = 0 # 0 = Flat, 1 = Long Ratio, -1 = Short Ratio
        entry_price = 0
        balance = capital
        equity_curve = [capital]
        timestamps = [df.index[0]]
        
        # Iterate row by row (Simulating live stream)
        for i, (index, row) in enumerate(df.iterrows()):
            
            z = row['Z-Score']
            current_ratio = row['Ratio']
            
            # --- ENTRY LOGIC ---
            # If Z-Score is very low (e.g. -1.5), Ratio is "cheap" -> BUY ETH, SELL BTC
            if position == 0 and z < -threshold:
                position = 1 
                entry_price = current_ratio
                trades.append({
                    'Time': index, 'Type': 'OPEN LONG', 'Price': current_ratio, 'Z': z, 'PnL': 0
                })
                
            # If Z-Score is very high (e.g. +1.5), Ratio is "expensive" -> SELL ETH, BUY BTC
            elif position == 0 and z > threshold:
                position = -1
                entry_price = current_ratio
                trades.append({
                    'Time': index, 'Type': 'OPEN SHORT', 'Price': current_ratio, 'Z': z, 'PnL': 0
                })

            # --- EXIT LOGIC (Mean Reversion) ---
            # If Long and Z-Score returns to 0 (or crosses above)
            elif position == 1 and z >= 0:
                # Profit Calculation: (Exit - Entry) / Entry
                raw_pnl = (current_ratio - entry_price) / entry_price
                net_pnl = raw_pnl - (fee * 2) # Entry fee + Exit fee
                
                profit_cash = balance * net_pnl
                balance += profit_cash
                
                position = 0
                trades.append({
                    'Time': index, 'Type': 'CLOSE LONG', 'Price': current_ratio, 'Z': z, 'PnL': profit_cash
                })

            # If Short and Z-Score returns to 0 (or crosses below)
            elif position == -1 and z <= 0:
                # Profit Calculation for Short: (Entry - Exit) / Entry
                raw_pnl = (entry_price - current_ratio) / entry_price
                net_pnl = raw_pnl - (fee * 2)
                
                profit_cash = balance * net_pnl
                balance += profit_cash
                
                position = 0
                trades.append({
                    'Time': index, 'Type': 'CLOSE SHORT', 'Price': current_ratio, 'Z': z, 'PnL': profit_cash
                })
            
            equity_curve.append(balance)
            timestamps.append(index)

        # --- 4. VISUALIZATION ---
        
        # Calculate Stats
        trade_df = pd.DataFrame(trades)
        total_trades = len(trade_df)
        days = (df.index[-1] - df.index[0]).days
        days = 1 if days < 1 else days
        trades_per_day = total_trades / days
        
        # METRICS ROW
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-container"><div class="big-num">{total_trades}</div><div class="label">Total Trades</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-container"><div class="big-num">{trades_per_day:.1f}</div><div class="label">Trades / Day</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-container"><div class="big-num" style="color:{"#00ff41" if balance >= capital else "#ff4b4b"}">${balance:,.2f}</div><div class="label">Final Balance</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-container"><div class="big-num">{len(df)}</div><div class="label">Datapoints Processed</div></div>', unsafe_allow_html=True)

        st.write("---")

        # PLOT 1: Z-Score & Signals
        st.subheader("1. The Signal: Z-Score")
        st.caption("We trade when the purple line exits the green zone. Notice how often it spikes?")
        
        fig_z = go.Figure()
        
        # Z-Score Line
        fig_z.add_trace(go.Scatter(x=df.index, y=df['Z-Score'], name='Z-Score', line=dict(color='#a32eff', width=1.5)))
        
        # Threshold Bands
        fig_z.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Sell Zone")
        fig_z.add_hline(y=-threshold, line_dash="dash", line_color="green", annotation_text="Buy Zone")
        fig_z.add_hline(y=0, line_color="#555", line_width=1)
        
        # Background Safe Zone
        fig_z.add_hrect(y0=-threshold, y1=threshold, line_width=0, fillcolor="gray", opacity=0.1)
        
        st.plotly_chart(fig_z, use_container_width=True)

        # PLOT 2: Trade Log
        st.subheader("2. Trade Ledger")
        if not trade_df.empty:
            # Highlight positive PnL green, negative red
            def color_pnl(val):
                if val > 0: return 'color: #00ff41'
                if val < 0: return 'color: #ff4b4b'
                return ''
            
            st.dataframe(
                trade_df.style.map(color_pnl, subset=['PnL']),
                use_container_width=True,
                height=300
            )
        else:
            st.warning("No trades triggered. Try lowering the Threshold or Window settings.")

else:
    st.info("👋 Click 'Run Simulation' to fetch live data and start the HFT logic.")
    
    # Educational Schematic
    st.markdown("#### How this creates High Frequency:")
    st.markdown("""
    1. **Data:** We pull 1-minute candles (e.g., 10,000 datapoints/week).
    2. **Logic:** We calculate the **Spread** (ETH price / BTC price).
    3. **Trigger:** We assume this ratio is "elastic."
        * If ETH shoots up relative to BTC (Z-Score > 1.0), we **Short**.
        * If ETH crashes relative to BTC (Z-Score < -1.0), we **Long**.
    4. **Volume:** Because crypto is volatile, the Z-Score crosses 1.0 many times a day.
    """)
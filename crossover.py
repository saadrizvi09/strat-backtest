import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- Configuration ---
st.set_page_config(page_title="Regime Filtered Backtester", layout="wide")
st.title("🧠 Smart Regime-Filtered Strategy")
st.markdown("This strategy uses **SuperTrend for Direction** but uses **Volatility (ATR)** to decide if the market is too dangerous to trade.")

# --- Sidebar ---
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value="BTC-USD")
start_date = st.sidebar.date_input("Start", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End", value=datetime.now())

st.sidebar.subheader("SuperTrend (Direction)")
atr_period = st.sidebar.slider("ATR Period", 5, 50, 10)
multiplier = st.sidebar.slider("ATR Multiplier", 1.0, 10.0, 3.0)

st.sidebar.subheader("Regime Filter (Risk)")
# Calculate Rolling Volatility percentile
vol_lookback = st.sidebar.slider("Regime Lookback (Days)", 30, 365, 100)
vol_threshold = st.sidebar.slider("Panic Threshold (Percentile)", 50, 99, 90)
st.sidebar.caption("If current Volatility is in the top 10% (90th percentile) of the last 100 days, the bot goes to CASH.")

fee_pct = st.sidebar.number_input("Fee (%)", 0.0, 2.0, 0.1) / 100

# --- Functions ---
@st.cache_data
def get_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df = df.xs(symbol, level=1, axis=1)
        return df
    except: return None

def calculate_supertrend(df, period, multiplier):
    # (Same SuperTrend Logic as before - abbreviated for brevity)
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(period).mean()
    df['Basic_Upper'] = (df['High'] + df['Low']) / 2 + (multiplier * df['ATR'])
    df['Basic_Lower'] = (df['High'] + df['Low']) / 2 - (multiplier * df['ATR'])
    df['Final_Upper'] = df['Basic_Upper']
    df['Final_Lower'] = df['Basic_Lower']
    df['SuperTrend'] = 0.0
    
    for i in range(period, len(df)):
        if (df['Basic_Upper'].iloc[i] < df['Final_Upper'].iloc[i-1]) or (df['Close'].iloc[i-1] > df['Final_Upper'].iloc[i-1]):
            df.at[df.index[i], 'Final_Upper'] = df['Basic_Upper'].iloc[i]
        else:
            df.at[df.index[i], 'Final_Upper'] = df['Final_Upper'].iloc[i-1]
        
        if (df['Basic_Lower'].iloc[i] > df['Final_Lower'].iloc[i-1]) or (df['Close'].iloc[i-1] < df['Final_Lower'].iloc[i-1]):
            df.at[df.index[i], 'Final_Lower'] = df['Basic_Lower'].iloc[i]
        else:
            df.at[df.index[i], 'Final_Lower'] = df['Final_Lower'].iloc[i-1]
            
        if df['SuperTrend'].iloc[i-1] == df['Final_Upper'].iloc[i-1]:
            if df['Close'].iloc[i] > df['Final_Upper'].iloc[i]:
                df.at[df.index[i], 'SuperTrend'] = df['Final_Lower'].iloc[i]
            else:
                df.at[df.index[i], 'SuperTrend'] = df['Final_Upper'].iloc[i]
        else:
            if df['Close'].iloc[i] < df['Final_Lower'].iloc[i]:
                df.at[df.index[i], 'SuperTrend'] = df['Final_Upper'].iloc[i]
            else:
                df.at[df.index[i], 'SuperTrend'] = df['Final_Lower'].iloc[i]
    return df

# --- Main ---
raw_data = get_data(ticker, start_date, end_date)

if raw_data is not None and len(raw_data) > vol_lookback:
    data = raw_data.copy()
    data = calculate_supertrend(data, atr_period, multiplier)
    
    # 1. Direction Signal (SuperTrend)
    data['Trend_Up'] = np.where(data['SuperTrend'] == data['Final_Lower'], 1, 0)
    
    # 2. Regime Filter (The "Simple GARCH" alternative)
    # Calculate relative volatility (ATR / Price) to normalize it
    data['Norm_ATR'] = data['ATR'] / data['Close']
    
    # Calculate the rolling threshold (e.g., 90th percentile of volatility over last 100 days)
    data['Vol_Threshold'] = data['Norm_ATR'].rolling(window=vol_lookback).quantile(vol_threshold/100)
    
    # Regime Signal: 1 = Safe, 0 = Panic
    data['Safe_Regime'] = np.where(data['Norm_ATR'] < data['Vol_Threshold'], 1, 0)
    
    # 3. Final Position = Trend * Regime
    # We only buy if Trend is UP AND Regime is SAFE
    data['Signal'] = data['Trend_Up'] * data['Safe_Regime']
    data['Position'] = data['Signal'].shift(1)
    
    # Returns
    data['Market_Ret'] = data['Close'].pct_change()
    trades = data['Position'].diff().abs()
    data['Strat_Ret'] = (data['Market_Ret'] * data['Position']) - (trades * fee_pct)
    
    data['Cum_Market'] = (1 + data['Market_Ret']).cumprod()
    data['Cum_Strat'] = (1 + data['Strat_Ret'].fillna(0)).cumprod()
    
    # Results
    strat_res = data['Cum_Strat'].iloc[-1] - 1
    mark_res = data['Cum_Market'].iloc[-1] - 1
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Regime Strategy", f"{strat_res:.2%}", delta=f"{strat_res-mark_res:.2%}")
    c2.metric("Buy & Hold", f"{mark_res:.2%}")
    c3.metric("Trades Skipped (Too Risky)", int(len(data[(data['Trend_Up']==1) & (data['Safe_Regime']==0)])))

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Cum_Strat'], name='Smart Strategy', line=dict(color='#00CC96', width=3)))
    fig.add_trace(go.Scatter(x=data.index, y=data['Cum_Market'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
    
    # Add Red zones for Panic Regimes
    # (Visualizing where the bot refused to trade)
    panic_zones = data[data['Safe_Regime'] == 0]
    fig.add_trace(go.Scatter(x=panic_zones.index, y=data.loc[panic_zones.index, 'Cum_Market'], mode='markers', name='Panic Zone (No Trade)', marker=dict(color='red', size=2)))

    st.plotly_chart(fig, use_container_width=True)
    
    st.info("**How to Read:** The Red Dots on the Buy & Hold line indicate days where the Strategy refused to buy (even if price was going up) because Volatility was too high. This often filters out 'Blow-off tops' before a crash.")
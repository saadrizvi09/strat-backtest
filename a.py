import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import streamlit.components.v1 as components
from bokeh.embed import file_html
from bokeh.resources import CDN

# --- Page Config ---
st.set_page_config(page_title="Sniper Headshot Backtester", layout="wide")

# --- Helper Functions ---
def get_data(ticker, start, end, interval):
    try:
        # auto_adjust=True fixes split/dividend data issues
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
        
        if df.empty:
            st.error(f"❌ No data found for {ticker}. specific 15m data is limited to 60 days.")
            return None

        # Fix MultiIndex columns (YFinance bug workaround)
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            elif 'Close' in df.columns.get_level_values(1):
                df.columns = df.columns.get_level_values(1)
            
        required = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required):
            st.error("❌ Data missing required Open/High/Low/Close columns.")
            return None

        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_buy_hold_stats(df):
    if df.empty: return 0, 0
    first_price = df['Open'].iloc[0]
    last_price = df['Close'].iloc[-1]
    buy_hold_return = ((last_price - first_price) / first_price) * 100
    
    df['BH_CumMax'] = df['Close'].cummax()
    df['BH_Drawdown'] = (df['Close'] - df['BH_CumMax']) / df['BH_CumMax']
    max_dd = df['BH_Drawdown'].min() * 100
    return buy_hold_return, max_dd

# --- Strategy Class ---
class SniperHeadshotStrategy(Strategy):
    swing_lookback = 15     
    risk_reward = 2.5       
    
    def init(self):
        # Strategy initialized
        pass
    
    def next(self):
        # 1. Check Data Availability
        if len(self.data) <= self.swing_lookback: return

        # 2. Get Data
        current_low = self.data.Low[-1]
        current_high = self.data.High[-1]
        current_open = self.data.Open[-1]
        current_close = self.data.Close[-1]
        
        recent_lows = self.data.Low[-self.swing_lookback:-1]
        recent_highs = self.data.High[-self.swing_lookback:-1]
        
        lowest_support = np.min(recent_lows)
        highest_resistance = np.max(recent_highs)

        # 3. Define Signals
        is_green_candle = current_close > current_open
        is_red_candle = current_close < current_open
        
        liquidity_sweep_low = current_low < lowest_support 
        liquidity_sweep_high = current_high > highest_resistance 

        # 4. BUY SETUP (Long)
        if liquidity_sweep_low and is_green_candle:
            if not self.position.is_long:
                sl_price = current_low
                risk = current_close - sl_price
                
                if risk > 0:
                    tp_price = current_close + (risk * self.risk_reward)
                    
                    # Close Short if exists
                    if self.position.is_short:
                        self.position.close()
                    
                    # Size = 0.90 (90% of Equity). 
                    # Thanks to margin=0.05 (20x leverage), this will NEVER fail.
                    self.buy(sl=sl_price, tp=tp_price, size=0.90)

        # 5. SELL SETUP (Short)
        elif liquidity_sweep_high and is_red_candle:
            if not self.position.is_short:
                sl_price = current_high
                risk = sl_price - current_close
                
                if risk > 0:
                    tp_price = current_close - (risk * self.risk_reward)
                    
                    # Close Long if exists
                    if self.position.is_long:
                        self.position.close()
                    
                    # Size = 0.90 (90% of Equity)
                    self.sell(sl=sl_price, tp=tp_price, size=0.90)

# --- Streamlit UI ---
st.title("🎯 AlgoQuant: Sniper Headshot Backtester")
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Ticker", value="BTC-USD")
st.sidebar.info("⚠️ Start Date must be < 60 days ago for 15m/5m timeframe.")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("today") - pd.Timedelta(days=59))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
timeframe = st.sidebar.selectbox("Timeframe", ["15m", "30m", "1h", "1d"], index=0)
initial_cash = st.sidebar.number_input("Capital", value=10000)

if st.sidebar.button("Run Backtest"):
    with st.spinner("Running Backtest..."):
        df = get_data(ticker, start_date, end_date, timeframe)
        
        if df is not None:
            if len(df) < 50:
                st.warning("⚠️ Warning: Data sample is small. Results may vary.")

            # --- CRITICAL FIX: LEVERAGE ADDED ---
            # margin=0.05 means 20x Leverage. 
            # This virtually guarantees your 'size=0.90' orders will always be accepted.
            bt = Backtest(
                df, 
                SniperHeadshotStrategy, 
                cash=initial_cash, 
                commission=0.0,   # Set to 0 to keep math simple
                margin=0.05,      # <--- THE FIX: 20x Leverage
                exclusive_orders=False, 
                trade_on_close=True
            )
            
            stats = bt.run()
            bh_return, bh_dd = calculate_buy_hold_stats(df)
            
            # --- Results ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Strategy Return", f"{stats['Return [%]']:.2f}%")
            col2.metric("Total Trades", stats['# Trades'])
            col3.metric("Buy & Hold Return", f"{bh_return:.2f}%")
            
            st.divider()
            st.subheader("Comparison Table")
            
            comparison_data = {
                "Metric": ["Total Return", "Max Drawdown", "Win Rate", "Trades"],
                "Sniper Strategy": [
                    f"{stats['Return [%]']:.2f}%", 
                    f"{stats['Max. Drawdown [%]']:.2f}%", 
                    f"{stats['Win Rate [%]']:.2f}%", 
                    str(stats['# Trades'])
                ],
                "Buy & Hold": [
                    f"{bh_return:.2f}%", 
                    f"{bh_dd:.2f}%", 
                    "N/A", 
                    "1"
                ]
            }
            st.table(pd.DataFrame(comparison_data))
            
            # --- Plotting ---
            st.subheader("Equity Curve & Trade Visualization")
            try:
                p = bt.plot(open_browser=False, resample=False)
                html = file_html(p, CDN, "Sniper Strategy Plot")
                components.html(html, height=800, scrolling=True)
            except Exception as e:
                st.error(f"Could not render plot: {e}")
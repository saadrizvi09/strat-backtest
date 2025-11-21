import streamlit as st
import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sniper Algo (High Win Rate)", layout="wide", page_icon="🎯")

# --- CUSTOM INDICATOR: SAFE RSI (Manual Math Fix) ---
class SafeRSI(bt.Indicator):
    lines = ('rsi',)
    params = (('period', 14), ('movav', bt.indicators.MovAv.SMA))

    def __init__(self):
        # 1. Calculate Price Change (Close - Previous Close)
        change = self.data - self.data(-1)
        
        # 2. Separate Gains (Up) and Losses (Down)
        # If change > 0, Up = change. If change < 0, Up = 0.
        up = bt.indicators.Max(change, 0.0)
        
        # If change < 0, Down = -change (positive value). If change > 0, Down = 0.
        down = -bt.indicators.Min(change, 0.0)
        
        # 3. Calculate Averages
        maup = self.p.movav(up, period=self.p.period)
        madown = self.p.movav(down, period=self.p.period)
        
        # 4. CRITICAL FIX: Ensure denominator is never 0 (Math Safety)
        madown_safe = bt.indicators.Max(madown, 0.0000001)

        # 5. RSI Formula
        rs = maup / madown_safe
        self.lines.rsi = 100.0 - (100.0 / (1.0 + rs))

# --- STRATEGY CLASS (Sniper) ---
class SniperStrategy(bt.Strategy):
    params = (
        ('sma_period', 200),      # Trend Filter
        ('rsi_period', 2),        # Super Fast RSI
        ('rsi_entry', 10),        # Buy when Panic (RSI < 10)
        ('rsi_exit', 70),         # Sell when bounce (RSI > 70)
        ('max_exposure', 0.99),   # Use 99% of cash
        ('trading_start_date', None),
    )

    def __init__(self):
        self.sma200 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_period)
        # USE SAFE RSI HERE
        self.rsi = SafeRSI(self.data.close, period=self.p.rsi_period)
        self.trade_history = [] 

    def notify_trade(self, trade):
        if not trade.isclosed: return
        
        # Calculate PnL
        pnl = trade.pnlcomm
        current_val = self.broker.getvalue()
        pnl_pct = (pnl / current_val) * 100 if current_val > 0 else 0

        # Robust Exit Price Calculation
        if len(trade.history) > 0:
            initial_size = trade.history[0].event.size
            if initial_size != 0:
                exit_price = trade.price + (trade.pnlcomm / initial_size)
            else:
                exit_price = trade.price
        else:
            exit_price = self.data.close[0]

        # Log to Session State
        if 'trades' in st.session_state:
            st.session_state.trades.append({
                'Symbol': st.session_state.ticker,
                'Type': 'LONG',
                'Entry Price': trade.price,
                'Exit Price': exit_price,
                'PnL': trade.pnlcomm,
                'Return %': pnl_pct
            })

    def next(self):
        # Date Filter
        if self.p.trading_start_date:
            if self.data.datetime.date(0) < self.p.trading_start_date: return

        # Warmup
        if len(self) < self.p.sma_period: return
        
        price = self.data.close[0]
        
        # Record Equity
        if 'equity_curve' in st.session_state:
            st.session_state.equity_curve.append({
                'Date': self.data.datetime.date(0),
                'Equity': self.broker.getvalue()
            })

        # --- LOGIC ---
        
        # 1. EXIT LOGIC (Take Profit Quickly)
        if self.position:
            if self.rsi[0] > self.p.rsi_exit:
                self.close()
        
        # 2. ENTRY LOGIC (Buy the Fear)
        else:
            # Rule 1: Must be in Bull Market (Price > 200 SMA)
            # Rule 2: Must be Oversold (RSI 2 < 10)
            if price > self.sma200[0] and self.rsi[0] < self.p.rsi_entry:
                self.buy_all_in()

    def buy_all_in(self):
        cash = self.broker.get_cash()
        size = int((cash * self.p.max_exposure) / self.data.close[0])
        if size > 0:
            self.buy(size=size)

# --- STREAMLIT UI ---
st.title("🎯 Sniper Bot: High Win Rate Strategy")
st.markdown("""
**Logic:** Larry Connors RSI-2 Strategy.
* **High Probability:** Only buys deep pullbacks in a strong uptrend.
* **Quick Exits:** Takes profit immediately on the bounce.
* **Goal:** 80-90% Win Rate.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    asset_class = st.radio("Asset Class", ["Crypto", "US Stocks"])
    
    if asset_class == "Crypto":
        options = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD"]
    else:
        options = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "AMD", "QQQ", "SPY", "TQQQ"]
        
    ticker = st.selectbox("Select Asset", options)
    
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = col2.date_input("End Date", datetime.date(2023, 12, 31))
    
    st.subheader("Settings")
    rsi_entry = st.slider("RSI Entry Threshold (Lower = Safer)", 2, 20, 10)
    
    run_btn = st.button("🎯 Run Sniper", type="primary")

def get_data(ticker, start, end):
    warmup_start = start - datetime.timedelta(days=300)
    try:
        df = yf.download(ticker, start=warmup_start, end=end, interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

if run_btn:
    with st.spinner(f"Sniping {ticker}..."):
        st.session_state.equity_curve = [] 
        st.session_state.trades = [] 
        st.session_state.broker_val = 100000 
        st.session_state.ticker = ticker 

        df = get_data(ticker, start_date, end_date)
        
        if not df.empty:
            # Calculate Buy & Hold
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            bh_df = df.loc[mask]
            
            if len(bh_df) > 0:
                start_p = bh_df['Close'].iloc[0]
                end_p = bh_df['Close'].iloc[-1]
                bh_ret = ((end_p - start_p) / start_p) * 100
                bh_val = 100000 * (1 + bh_ret/100)
                
                # Run Backtest
                cerebro = bt.Cerebro()
                cerebro.addstrategy(SniperStrategy, 
                                  rsi_entry=rsi_entry,
                                  trading_start_date=start_date)
                
                cerebro.adddata(bt.feeds.PandasData(dataname=df))
                cerebro.broker.setcash(100000)
                
                # Run without runonce to avoid index errors with custom indicators
                results = cerebro.run(runonce=False)
                strat = results[0]
                
                final_val = cerebro.broker.getvalue()
                bot_ret = ((final_val / 100000) - 1) * 100
                
                # --- RESULTS ---
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Starting Capital", "$100,000")
                c2.metric("Final Value", f"${final_val:,.0f}", f"{bot_ret:.1f}%")
                c3.metric("Buy & Hold", f"${bh_val:,.0f}", f"{bh_ret:.1f}%")
                
                # Win Rate Calc
                trades_df = pd.DataFrame(st.session_state.trades)
                if not trades_df.empty:
                    win_rate = len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100
                    st.caption(f"Win Rate: {win_rate:.1f}% | Total Trades: {len(trades_df)}")
                
                # --- CHARTING ---
                st.subheader("📈 Equity Curve")
                
                eq_df = pd.DataFrame(st.session_state.equity_curve)
                if not eq_df.empty:
                    eq_df = eq_df[eq_df['Date'] >= start_date]
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=eq_df['Date'], y=eq_df['Equity'], name="Bot Equity", line=dict(color="#00FF00", width=2)), secondary_y=False)
                    fig.add_trace(go.Scatter(x=bh_df.index, y=bh_df['Close'], name=f"{ticker} Price", line=dict(color="gray", dash='dot')), secondary_y=True)
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- TRADE LOG ---
                st.subheader("📋 Trade Ledger")
                if not trades_df.empty:
                    def color_pnl(val):
                        color = '#90EE90' if val > 0 else '#FFCCCB'
                        return f'background-color: {color}; color: black'
                    st.dataframe(trades_df.style.map(color_pnl, subset=['PnL']), use_container_width=True)
                else:
                    st.info("No trades triggered.")
            else:
                st.error("Not enough data.")
        else:
            st.error("Data download failed.")

# --- LIVE SCANNER ---
st.sidebar.divider()
st.sidebar.header("📡 Panic Scanner")
if st.sidebar.button("Find Panic Dips"):
    st.sidebar.info("Scanning for RSI < 10 opportunities...")
    scan_list = ["BTC-USD", "ETH-USD", "NVDA", "TSLA", "QQQ"]
    for c in scan_list:
        try:
            df = yf.download(c, period="6mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # Calc Indicators Manually for Scanner
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
            
            # Manual Safe Division for Scanner
            rs = gain / loss.replace(0, 0.000001)
            rsi = 100 - (100 / (1 + rs))
            
            sma = df['Close'].rolling(200).mean()
            
            last_rsi = rsi.iloc[-1]
            last_price = df['Close'].iloc[-1]
            last_sma = sma.iloc[-1]
            
            if pd.notna(last_sma) and last_price > last_sma and last_rsi < 10:
                st.sidebar.success(f"{c}: BUY SIGNAL! (RSI {last_rsi:.1f})")
            else:
                st.sidebar.write(f"{c}: Wait (RSI {last_rsi:.1f})")
        except: pass
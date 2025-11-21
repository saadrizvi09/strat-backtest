import streamlit as st
import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Dynamic Regime Switcher", layout="wide", page_icon="🦁")

# --- CUSTOM INDICATOR: SAFE RSI ---
class SafeRSI(bt.Indicator):
    lines = ('rsi',)
    params = (('period', 14), ('movav', bt.indicators.MovAv.SMA))

    def __init__(self):
        change = self.data - self.data(-1)
        up = bt.indicators.Max(change, 0.0)
        down = -bt.indicators.Min(change, 0.0)
        maup = self.p.movav(up, period=self.p.period)
        madown = self.p.movav(down, period=self.p.period)
        madown_safe = bt.indicators.Max(madown, 0.0000001)
        rs = maup / madown_safe
        self.lines.rsi = 100.0 - (100.0 / (1.0 + rs))

# --- STRATEGY: DYNAMIC REGIME SWITCHER ---
class RegimeSwitcher(bt.Strategy):
    params = (
        ('rsi_period', 2),
        ('rsi_entry', 10),
        ('rsi_exit', 70),
        ('trend_sma', 50),
        ('safety_sma', 200),
        ('adx_period', 14),
        ('adx_threshold', 25),  # The Switch Point
        ('stop_loss', 0.05),    # 5% Hard Stop
        ('max_exposure', 0.98), # 98% Capital Usage
        ('trading_start_date', None),
    )

    def __init__(self):
        # Indicators
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_period)
        self.rsi = SafeRSI(self.data.close, period=self.p.rsi_period)
        self.sma50 = bt.indicators.SMA(self.data.close, period=self.p.trend_sma)
        self.sma200 = bt.indicators.SMA(self.data.close, period=self.p.safety_sma)
        
        # State tracking
        self.regime = "Unknown" 
        self.entry_regime = "" # Remembers which logic bought the asset

    def notify_trade(self, trade):
        if not trade.isclosed: return
        
        pnl = trade.pnlcomm
        current_val = self.broker.getvalue()
        pnl_pct = (pnl / current_val) * 100 if current_val > 0 else 0
        
        entry_dt = bt.num2date(trade.dtopen)
        exit_dt = bt.num2date(trade.dtclose)

        if 'trades' in st.session_state:
            st.session_state.trades.append({
                'Symbol': st.session_state.ticker,
                'Type': 'LONG',
                'Entry Date': entry_dt.strftime('%Y-%m-%d'),
                'Exit Date': exit_dt.strftime('%Y-%m-%d'),
                'Entry Price': trade.price,
                'Exit Price': trade.price + (pnl/trade.size) if trade.size!=0 else trade.price,
                'PnL': pnl,
                'Return %': pnl_pct,
                'Regime': self.entry_regime # Logs "Bull Trend" or "Sniper Chop"
            })

    def next(self):
        # Date Filter
        if self.p.trading_start_date:
            if self.data.datetime.date(0) < self.p.trading_start_date: return
        
        # Warmup
        if len(self) < 200: return

        current_price = self.data.close[0]
        
        # Record Equity
        if 'equity_curve' in st.session_state:
            st.session_state.equity_curve.append({
                'Date': self.data.datetime.date(0),
                'Equity': self.broker.getvalue()
            })

        # --- 1. EMERGENCY STOP LOSS (Always First) ---
        if self.position:
            pnl_pct = (current_price - self.position.price) / self.position.price
            if pnl_pct < -self.p.stop_loss:
                self.close()
                return # Exit immediately

        # --- 2. DETERMINE REGIME ---
        # ADX > 25 = Strong Trend (Bull Mode)
        # ADX < 25 = Choppy/Sideways (Sniper Mode)
        is_trending = self.adx[0] > self.p.adx_threshold

        # --- 3. ENTRY LOGIC ---
        if not self.position:
            if is_trending:
                # >>> BULL MODE <<<
                # Buy if Price > SMA 50 (Trend Following)
                if current_price > self.sma50[0]:
                    self.buy_all_in()
                    self.entry_regime = "Bull Trend"
            
            else:
                # >>> SNIPER MODE <<<
                # Buy if Price > SMA 200 (Safety) AND RSI < 10 (Panic)
                if current_price > self.sma200[0] and self.rsi[0] < self.p.rsi_entry:
                    self.buy_all_in()
                    self.entry_regime = "Sniper Chop"

        # --- 4. EXIT LOGIC ---
        else:
            # Exit logic depends on how we entered
            if self.entry_regime == "Bull Trend":
                # Trend Exit: Hold until price breaks BELOW SMA 50
                if current_price < self.sma50[0]:
                    self.close()
            
            elif self.entry_regime == "Sniper Chop":
                # Sniper Exit: Quick profit at RSI > 70
                if self.rsi[0] > self.p.rsi_exit:
                    self.close()

    def buy_all_in(self):
        cash = self.broker.get_cash()
        size = int((cash * self.p.max_exposure) / self.data.close[0])
        if size > 0:
            self.buy(size=size)

# --- STREAMLIT UI ---
st.title("🦁 Dynamic Bot: Auto-Switching Logic")
st.markdown("""
**Strategy:** Smart Regime Switching (ADX Filter)
* **Choppy Market (ADX < 25):** Uses **Sniper Mode** (RSI Mean Reversion).
* **Trending Market (ADX > 25):** Uses **Bull Mode** (Trend Following).
* **Safety:** Hard 5% Stop Loss on all trades.
""")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    asset_class = st.radio("Asset Class", ["Crypto", "US Stocks"])
    
    if asset_class == "Crypto":
        options = ["Use Custom Ticker", "SOL-USD", "BTC-USD", "ETH-USD", "DOGE-USD", "PEPE-USD", "WIF-USD"]
    else:
        options = ["Use Custom Ticker", "NVDA", "TSLA", "MSTR", "COIN", "AMD"]
        
    selected_option = st.selectbox("Select Asset", options, index=1)
    if selected_option == "Use Custom Ticker":
        ticker = st.text_input("Ticker", value="SOL-USD").upper()
    else:
        ticker = selected_option
    
    st.divider()
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start", datetime.date(2024, 1, 1))
    end_date = col2.date_input("End", datetime.date(2025, 12, 31))
    
    st.subheader("Tuning")
    adx_thresh = st.slider("Trend Definition (ADX)", 15, 40, 25, help="Higher = Harder to switch to Trend Mode")
    rsi_entry = st.slider("Sniper Entry (RSI)", 2, 15, 10)
    
    run_btn = st.button("🚀 Run Dynamic Bot", type="primary")

def get_data(ticker, start, end):
    warmup_start = start - datetime.timedelta(days=300)
    try:
        df = yf.download(ticker, start=warmup_start, end=end, interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except: return pd.DataFrame()

# --- MAIN EXECUTION ---
if run_btn:
    with st.spinner(f"Analyzing {ticker} Market Regimes..."):
        st.session_state.equity_curve = [] 
        st.session_state.trades = [] 
        st.session_state.ticker = ticker 

        df = get_data(ticker, start_date, end_date)
        
        if not df.empty and len(df) > 200:
            # Buy & Hold
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            bh_df = df.loc[mask]
            start_p = bh_df['Close'].iloc[0]
            end_p = bh_df['Close'].iloc[-1]
            bh_ret = ((end_p - start_p) / start_p) * 100
            bh_val = 100000 * (1 + bh_ret/100)
            
            # Run Backtest
            cerebro = bt.Cerebro()
            cerebro.addstrategy(RegimeSwitcher, 
                                rsi_entry=rsi_entry,
                                adx_threshold=adx_thresh,
                                trading_start_date=start_date)
            
            cerebro.adddata(bt.feeds.PandasData(dataname=df))
            cerebro.broker.setcash(100000)
            cerebro.run()
            
            final_val = cerebro.broker.getvalue()
            bot_ret = ((final_val / 100000) - 1) * 100
            
            # --- METRICS ---
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Starting Capital", "$100,000")
            c2.metric("Final Value", f"${final_val:,.0f}", f"{bot_ret:.1f}%")
            c3.metric("Buy & Hold", f"${bh_val:,.0f}", f"{bh_ret:.1f}%")
            
            # --- EQUITY CHART ---
            st.subheader("📈 Equity Curve")
            eq_df = pd.DataFrame(st.session_state.equity_curve)
            if not eq_df.empty:
                eq_df = eq_df[eq_df['Date'] >= start_date]
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=eq_df['Date'], y=eq_df['Equity'], name="Dynamic Bot", line=dict(color="#00FF00", width=2)), secondary_y=False)
                fig.add_trace(go.Scatter(x=bh_df.index, y=bh_df['Close'], name="Price", line=dict(color="gray", dash='dot')), secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)

            # --- TRADE LEDGER ---
            st.subheader("📋 Trade Ledger")
            trades_df = pd.DataFrame(st.session_state.trades)
            if not trades_df.empty:
                st.dataframe(trades_df.style.map(
                    lambda x: f'background-color: {"#90EE90" if x > 0 else "#FFCCCB"}; color: black', 
                    subset=['PnL']
                ), use_container_width=True)
            else:
                st.info("No trades triggered.")
        else:
            st.error("Not enough data.")

# --- PANIC SCANNER ---
st.sidebar.divider()
if st.sidebar.button("📡 Scan for Dips"):
    st.sidebar.info("Scanning...")
    scan_list = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "NVDA", "MSTR"]
    for c in scan_list:
        try:
            df_scan = yf.download(c, period="3mo", progress=False)
            if isinstance(df_scan.columns, pd.MultiIndex): df_scan.columns = df_scan.columns.get_level_values(0)
            
            # Quick RSI Calc
            delta = df_scan['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
            rs = gain / loss.replace(0, 0.000001)
            rsi = 100 - (100 / (1 + rs))
            
            if rsi.iloc[-1] < 15: st.sidebar.error(f"🚨 {c}: RSI {rsi.iloc[-1]:.1f}")
            elif rsi.iloc[-1] < 30: st.sidebar.warning(f"👀 {c}: RSI {rsi.iloc[-1]:.1f}")
        except: pass
import streamlit as st
import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Regime Moonshot Algo", layout="wide", page_icon="🚀")

# --- STRATEGY CLASS (Your "Pro" Logic) ---
class RegimePolymath(bt.Strategy):
    params = (
        ('ema_fast', 13), ('ema_slow', 34), ('ema_trend', 55),
        ('rsi_period', 14), ('regime_sma', 200), ('regime_adx', 20),
        ('bull_risk', 0.03), ('bull_trail_mult', 6.0),
        ('bear_risk', 0.02), ('bear_trail_mult', 3.0),
        ('max_pos_pct', 0.95), ('pyramid_enabled', True),
        ('trading_start_date', None), # <--- ADDED THIS (Was missing)
    )

    def __init__(self):
        self.sma200 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.regime_sma)
        self.adx = bt.indicators.ADX(self.data, period=14)
        self.ema_fast = bt.indicators.EMA(period=self.p.ema_fast)
        self.ema_slow = bt.indicators.EMA(period=self.p.ema_slow)
        self.ema_trend = bt.indicators.EMA(period=self.p.ema_trend)
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.atr = bt.indicators.ATR(period=14)
        
        self.entry_price = None
        self.extremum_price = None 
        self.stop_price = None
        self.pyramid_count = 0
        self.regime = "UNKNOWN"

    def notify_trade(self, trade):
        if not trade.isclosed: return
        # Log trade for the frontend
        exit_price = trade.price + (trade.pnlcomm / trade.size) if trade.size != 0 else trade.price
        
        # Ensure session state exists before appending (safety check)
        if 'trades' in st.session_state:
            st.session_state.trades.append({
                'Symbol': st.session_state.ticker,
                'Type': 'LONG' if trade.size > 0 else 'SHORT',
                'Entry Price': trade.price,
                'Exit Price': exit_price,
                'PnL': trade.pnlcomm,
                'Return %': (trade.pnlcomm / st.session_state.broker_val) * 100
            })

    def next(self):
        # Check if trading start date param is set and valid
        if self.p.trading_start_date:
            current_date = self.data.datetime.date(0)
            # If current data date is BEFORE user start date, skip logic
            if current_date < self.p.trading_start_date:
                return

        if len(self) < self.p.regime_sma: return
        price = self.data.close[0]
        
        # Regime Detection
        is_bullish = price > self.sma200[0]
        is_bearish = price < self.ema_trend[0]
        is_strong = self.adx[0] > self.p.regime_adx
        
        if is_bullish and is_strong: self.regime = "BULL"
        elif is_bearish and is_strong: self.regime = "BEAR"
        else: self.regime = "NEUTRAL"

        # Store data for plotting later
        if 'equity_curve' in st.session_state:
            date = self.data.datetime.date(0)
            st.session_state.equity_curve.append({'Date': date, 'Equity': self.broker.getvalue()})
        
        # Execution Logic
        if self.position:
            if self.position.size > 0: self.manage_long(price)
            else: self.manage_short(price)
        else:
            if self.regime == "BULL": self.find_long_entry(price)
            elif self.regime == "BEAR": self.find_short_entry(price)

    def find_long_entry(self, price):
        if self.ema_fast[0] > self.ema_slow[0] and self.rsi[0] < 70 and price > self.ema_trend[0]:
            self.enter_trade(price, is_long=True)

    def find_short_entry(self, price):
        if self.ema_fast[0] < self.ema_slow[0] and price < self.ema_trend[0]:
            self.enter_trade(price, is_long=False)

    def manage_long(self, price):
        atr = self.atr[0]
        if price > self.extremum_price: self.extremum_price = price
        new_stop = self.extremum_price - (self.p.bull_trail_mult * atr)
        if new_stop > self.stop_price: self.stop_price = new_stop
        if price < self.stop_price: self.close(); return

        if self.p.pyramid_enabled and self.pyramid_count < 3:
            pnl_pct = (price - self.entry_price) / self.entry_price
            if pnl_pct > 0.20 * (self.pyramid_count + 1) and self.rsi[0] < 80:
                self.pyramid_add(price, is_long=True)

    def manage_short(self, price):
        atr = self.atr[0]
        if price < self.extremum_price: self.extremum_price = price
        new_stop = self.extremum_price + (self.p.bear_trail_mult * atr)
        if new_stop < self.stop_price: self.stop_price = new_stop
        if price > self.stop_price: self.close(); return

        if self.p.pyramid_enabled and self.pyramid_count < 3:
            pnl_pct = (self.entry_price - price) / self.entry_price
            if pnl_pct > 0.15 * (self.pyramid_count + 1):
                self.pyramid_add(price, is_long=False)

    def enter_trade(self, price, is_long):
        atr = self.atr[0]
        risk_pct = self.p.bull_risk if is_long else self.p.bear_risk
        stop_mult = 3.0 if is_long else 2.0
        dist_to_stop = stop_mult * atr
        risk_usd = self.broker.getvalue() * risk_pct
        
        if dist_to_stop > 0:
            size = int(risk_usd / dist_to_stop)
            max_size = int((self.broker.getvalue() * self.p.max_pos_pct) / price)
            size = min(size, max_size)
            if size > 0:
                if is_long: self.buy(size=size); self.stop_price = price - dist_to_stop
                else: self.sell(size=size); self.stop_price = price + dist_to_stop
                self.entry_price = price; self.extremum_price = price; self.pyramid_count = 0

    def pyramid_add(self, price, is_long):
        cash = self.broker.get_cash()
        add_size = int((cash * 0.30) / price)
        atr = self.atr[0]
        if add_size > 0:
            if is_long: self.buy(size=add_size); self.stop_price = price - (2.0 * atr)
            else: self.sell(size=add_size); self.stop_price = price + (2.0 * atr)
            self.pyramid_count += 1

# --- STREAMLIT UI ---
st.title("🌖 Regime Moonshot: Algorithmic Backtester")
st.markdown("### Professional Trend Following & Regime Switching System")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    ticker = st.selectbox("Select Asset", 
        ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD", 
         "DOGE-USD", "LINK-USD", "LTC-USD", "BCH-USD", "UNI-USD"])
    
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = col2.date_input("End Date", datetime.date(2023, 1, 1))
    
    st.subheader("Strategy Risk")
    risk_per_trade = st.slider("Risk Per Trade (%)", 1, 10, 3) / 100
    enable_pyramid = st.checkbox("Enable Pyramiding (Compound Winners)", True)
    
    run_btn = st.button("🚀 Run Backtest", type="primary")

# Helper to download data
def get_data(ticker, start, end):
    # Download extra 300 days for warm-up
    warmup_start = start - datetime.timedelta(days=300)
    try:
        df = yf.download(ticker, start=warmup_start, end=end, interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Main Execution
if run_btn:
    with st.spinner(f"Running Regime Analysis on {ticker}..."):
        # Reset Session State for Reporting
        st.session_state.trades = []
        st.session_state.equity_curve = []
        st.session_state.ticker = ticker
        st.session_state.broker_val = 100000
        
        # 1. Get Data
        df = get_data(ticker, start_date, end_date)
        
        if not df.empty:
            # 2. Buy & Hold Calc (Only for selected period)
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            bh_df = df.loc[mask]
            if len(bh_df) > 0:
                start_price = bh_df['Close'].iloc[0]
                end_price = bh_df['Close'].iloc[-1]
                bh_return = (end_price - start_price) / start_price * 100
                
                # 3. Run Bot
                cerebro = bt.Cerebro()
                # FIX: Correctly map UI sliders to Strategy Params
                cerebro.addstrategy(RegimePolymath, 
                                  bull_risk=risk_per_trade,  # Map slider to Bull Risk
                                  bear_risk=risk_per_trade,  # Map slider to Bear Risk
                                  pyramid_enabled=enable_pyramid,
                                  trading_start_date=start_date)
                
                cerebro.adddata(bt.feeds.PandasData(dataname=df))
                cerebro.broker.setcash(100000)
                cerebro.broker.set_shortcash(False)
                
                cerebro.run()
                final_val = cerebro.broker.getvalue()
                bot_return = (final_val / 100000 - 1) * 100
                
                # --- RESULTS DASHBOARD ---
                st.divider()
                
                # KPI Cards
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Starting Capital", "$100,000")
                kpi2.metric("Final Value", f"${final_val:,.0f}", f"{bot_return:.1f}%")
                kpi3.metric("Buy & Hold", f"${100000 * (1 + bh_return/100):,.0f}", f"{bh_return:.1f}%")
                
                # Win Rate
                trades_df = pd.DataFrame(st.session_state.trades)
                if not trades_df.empty:
                    win_rate = len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100
                    st.caption(f"Win Rate: {win_rate:.1f}% | Total Trades: {len(trades_df)}")
                
                # Charting (Interactive Plotly)
                st.subheader("📈 Equity Curve vs Market")
                
                equity_df = pd.DataFrame(st.session_state.equity_curve)
                # Filter equity curve to user selection
                equity_df = equity_df[equity_df['Date'] >= start_date]
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Bot Equity
                fig.add_trace(go.Scatter(
                    x=equity_df['Date'], y=equity_df['Equity'], 
                    name="Bot Equity", line=dict(color="#00FF00", width=2)
                ), secondary_y=False)
                
                # Asset Price
                fig.add_trace(go.Scatter(
                    x=bh_df.index, y=bh_df['Close'], 
                    name=f"{ticker} Price", line=dict(color="gray", dash='dot')
                ), secondary_y=True)
                
                fig.update_layout(title_text="Performance Over Time", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade Log
                st.subheader("📋 Trade Log")
                if not trades_df.empty:
                    # Style the dataframe (Green wins, Red losses)
                    def color_pnl(val):
                        color = '#90EE90' if val > 0 else '#FFCCCB' # Light Green / Light Red
                        return f'background-color: {color}; color: black'
                        
                    st.dataframe(
                        trades_df.style.map(color_pnl, subset=['PnL']),
                        use_container_width=True
                    )
                else:
                    st.info("No trades were triggered in this period (Bot stayed safe).")
            else:
                st.error("Not enough data for selected period.")
        else:
            st.error("Failed to download data.")

# --- LIVE REGIME MONITOR ---
st.sidebar.divider()
st.sidebar.header("📡 Live Market Scanner")
if st.sidebar.button("Check Live Regimes"):
    st.sidebar.info("Scanning top coins...")
    coins = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
    
    for c in coins:
        try:
            live_df = yf.download(c, period="1y", interval="1d", progress=False)
            if isinstance(live_df.columns, pd.MultiIndex): live_df.columns = live_df.columns.get_level_values(0)
            
            # Quick Logic
            last_close = live_df['Close'].iloc[-1]
            sma200 = live_df['Close'].rolling(200).mean().iloc[-1]
            
            if last_close > sma200:
                st.sidebar.success(f"{c}: BULL 🐂")
            else:
                st.sidebar.error(f"{c}: BEAR 🐻")
        except:
            pass
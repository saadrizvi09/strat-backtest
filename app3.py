import streamlit as st
import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Meme Scaler Bot", layout="wide", page_icon="📈")

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

# --- STRATEGY: PURE SCALING OUT (2x, 4x, 8x) ---
class ScalingOutStrategy(bt.Strategy):
    params = (
        ('sma_period', 200),      
        ('rsi_period', 14),        
        ('rsi_entry', 30),        
        ('rsi_exit', 70),         
        ('stop_loss', 0.15),      
        ('max_exposure', 0.95),
        ('cooldown_period', 5),   
        ('trading_start_date', None),
        # THE SCALING CONFIGURATION
        ('targets', [2.0, 4.0, 8.0]),      # 2x, 4x, 8x Milestones
        ('portions', [0.25, 0.25, 0.25]),  # Sell 25% at each milestone
    )

    def __init__(self):
        self.sma200 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_period)
        self.rsi = SafeRSI(self.data.close, period=self.p.rsi_period)
        
        # Tracking Variables
        self.next_allowed_trade_bar = 0 
        self.entry_price = None
        self.initial_size = 0
        # Array to track which milestones we have already hit [False, False, False]
        self.milestones_hit = [False] * len(self.p.targets)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                # Capture Entry Data
                self.entry_price = order.executed.price
                self.initial_size = order.executed.size
                # Reset milestones for new trade
                self.milestones_hit = [False] * len(self.p.targets)

    def notify_trade(self, trade):
        if not trade.isclosed: return
        
        pnl = trade.pnlcomm
        # Cooldown if net result is loss
        if pnl < 0:
            self.next_allowed_trade_bar = len(self) + self.p.cooldown_period

        if 'trades' in st.session_state:
            st.session_state.trades.append({
                'Symbol': st.session_state.ticker,
                'Type': 'LONG',
                'Entry Date': bt.num2date(trade.dtopen).strftime('%Y-%m-%d'),
                'Exit Date': bt.num2date(trade.dtclose).strftime('%Y-%m-%d'),
                'Entry Price': trade.price, 
                # Calculate approximate average exit price based on PnL
                'Avg Exit Price': trade.price + (pnl / self.initial_size) if self.initial_size else trade.price, 
                'Total PnL': pnl,
                'Return %': (pnl / self.broker.getvalue()) * 100
            })

    def next(self):
        # Date & Warmup Filters
        if self.p.trading_start_date and self.data.datetime.date(0) < self.p.trading_start_date: return
        if len(self) < self.p.sma_period: return
        
        # Equity Curve
        if 'equity_curve' in st.session_state:
            st.session_state.equity_curve.append({
                'Date': self.data.datetime.date(0),
                'Equity': self.broker.getvalue()
            })

        # Check Cooldown
        if len(self) < self.next_allowed_trade_bar:
            return 

        # --- EXIT LOGIC ---
        if self.position:
            current_price = self.data.close[0]
            
            # 1. SAFETY: Hard Stop Loss
            stop_price = self.entry_price * (1.0 - self.p.stop_loss)
            if self.data.low[0] < stop_price:
                self.close() 
                return

            # 2. STRATEGY: Scaling Out (The Logic You Requested)
            # Iterate through our targets (2x, 4x, 8x)
            for i, target_mult in enumerate(self.p.targets):
                # If we haven't hit this specific milestone yet...
                if not self.milestones_hit[i]:
                    target_price = self.entry_price * target_mult
                    
                    # If price reached target...
                    if current_price >= target_price:
                        # Calculate sell size based on INITIAL position size
                        sell_amount = int(self.initial_size * self.p.portions[i])
                        
                        # Ensure we have enough to sell (safeguard)
                        if self.position.size >= sell_amount and sell_amount > 0:
                            self.sell(size=sell_amount)
                            self.milestones_hit[i] = True # Mark as hit so we don't sell again

            # 3. FINAL EXIT: RSI or Trend Break
            # This clears whatever is left (e.g., the last 25%)
            if self.rsi[0] > self.p.rsi_exit:
                self.close() 

        # --- ENTRY LOGIC ---
        else:
            if self.data.close[0] > self.sma200[0] and self.rsi[0] < self.p.rsi_entry:
                self.buy_all_in()

    def buy_all_in(self):
        cash = self.broker.get_cash()
        size = int((cash * self.p.max_exposure) / self.data.close[0])
        if size > 0:
            self.buy(size=size)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("📈 Scaling Out Config")
    
    asset_class = st.radio("Asset Class", ["Crypto", "US Stocks"])
    if asset_class == "Crypto":
        majors = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "XRP-USD"]
        memes = ["DOGE-USD", "SHIB-USD", "PEPE-USD", "WIF-USD", "BONK-USD", "FLOKI-USD"]
        options = ["Use Custom Ticker"] + majors + memes
    else:
        options = ["Use Custom Ticker", "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "MSTR", "COIN"]
        
    selected_option = st.selectbox("Select Asset", options, index=1)
    if selected_option == "Use Custom Ticker":
        ticker = st.text_input("Ticker Symbol", value="WIF-USD").upper()
    else:
        ticker = selected_option
    
    st.divider()
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start", datetime.date(2023, 1, 1))
    end_date = col2.date_input("End", datetime.date(2025, 12, 31))
    
    st.subheader("Milestones (Multipliers)")
    m1 = st.number_input("Target 1 (e.g., 2x)", value=2.0, step=0.5)
    m2 = st.number_input("Target 2 (e.g., 4x)", value=4.0, step=0.5)
    m3 = st.number_input("Target 3 (e.g., 8x)", value=8.0, step=1.0)
    
    st.subheader("Sell Percentage (Per Target)")
    pct_sell = st.slider("Sell % at each Target", 0.1, 0.3, 0.25, help="0.25 means sell 25% of initial bag")

    run_btn = st.button("Run Scaling Strategy", type="primary")

def get_data(ticker, start, end):
    warmup = start - datetime.timedelta(days=300)
    try:
        df = yf.download(ticker, start=warmup, end=end, interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except: return pd.DataFrame()

if run_btn:
    with st.spinner(f"Applying Scaling Strategy on {ticker}..."):
        st.session_state.equity_curve = [] 
        st.session_state.trades = [] 
        st.session_state.ticker = ticker
        
        df = get_data(ticker, start_date, end_date)
        
        if not df.empty:
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            bh_df = df.loc[mask]
            if not bh_df.empty:
                bh_val = 100000 * (1 + ((bh_df['Close'].iloc[-1] - bh_df['Close'].iloc[0])/bh_df['Close'].iloc[0]))
                
                cerebro = bt.Cerebro()
                # Pass the list of targets and the fixed percentage
                cerebro.addstrategy(ScalingOutStrategy, 
                                    targets=[m1, m2, m3],
                                    portions=[pct_sell, pct_sell, pct_sell],
                                    trading_start_date=start_date)
                
                cerebro.adddata(bt.feeds.PandasData(dataname=df))
                cerebro.broker.setcash(100000)
                cerebro.broker.setcommission(commission=0.001) 

                cerebro.run()
                
                final_val = cerebro.broker.getvalue()
                bot_ret = ((final_val / 100000) - 1) * 100
                
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Starting Capital", "$100,000")
                c2.metric("Final Value", f"${final_val:,.0f}", f"{bot_ret:.1f}%")
                c3.metric("Buy & Hold", f"${bh_val:,.0f}", f"{((bh_val/100000)-1)*100:.1f}%")
                
                eq_df = pd.DataFrame(st.session_state.equity_curve)
                if not eq_df.empty:
                    eq_df = eq_df[eq_df['Date'] >= start_date]
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=eq_df['Date'], y=eq_df['Equity'], name="Bot Equity", 
                                             line=dict(color="#00FF00", width=2)), secondary_y=False)
                    fig.add_trace(go.Scatter(x=bh_df.index, y=bh_df['Close'], name="Price", 
                                             line=dict(color="gray", dash='dot', width=1)), secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                trades_df = pd.DataFrame(st.session_state.trades)
                if not trades_df.empty:
                    st.subheader("Trade Ledger")
                    st.dataframe(trades_df.style.map(
                        lambda x: f'background-color: {"#90EE90" if x > 0 else "#FFCCCB"}; color: black', 
                        subset=['Total PnL']
                    ), use_container_width=True)
        else:
            st.error("Data download failed.")
import streamlit as st
import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="3-Down Sniper Strategy", layout="wide", page_icon="📉")

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

# --- STRATEGY: 3-DOWN + 1-UP CONFIRMATION ---
class ThreeDownStrategy(bt.Strategy):
    params = (
        ('sma_period', 200),      # Trend Filter
        ('rsi_period', 2),        # Used for EXIT only
        ('rsi_exit', 70),         # Profit Target
        ('stop_loss', 0.05),      # 5% Safety Net
        ('max_exposure', 0.98),   # 98% Capital
        ('trading_start_date', None),
    )

    def __init__(self):
        # Indicators
        self.sma200 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_period)
        self.rsi = SafeRSI(self.data.close, period=self.p.rsi_period)
        self.stop_order = None # Track the stop order

    def notify_order(self, order):
        # Handle the Automatic Stop Loss orders
        if order.status in [order.Completed]:
            if order.isbuy():
                # Buy Filled -> Place Stop Loss immediately
                stop_price = order.executed.price * (1.0 - self.p.stop_loss)
                self.stop_order = self.sell(exectype=bt.Order.Stop, price=stop_price)
            elif order.issell():
                # Sell Filled -> Stop Loss no longer needed
                self.stop_order = None

    def notify_trade(self, trade):
        if not trade.isclosed: return
        
        pnl = trade.pnlcomm
        entry_dt = bt.num2date(trade.dtopen)
        exit_dt = bt.num2date(trade.dtclose)
        
        # Calculate Real Exit Price (Fixes display bug)
        if len(trade.history) > 0:
            initial_size = trade.history[0].event.size
            price_diff = pnl / initial_size
            exit_price = trade.price + price_diff
        else:
            exit_price = trade.price

        if 'trades' in st.session_state:
            st.session_state.trades.append({
                'Symbol': st.session_state.ticker,
                'Type': 'LONG',
                'Entry Date': entry_dt.strftime('%Y-%m-%d'),
                'Exit Date': exit_dt.strftime('%Y-%m-%d'),
                'Entry Price': trade.price,
                'Exit Price': exit_price,
                'PnL': pnl,
                'Return %': (pnl / self.broker.getvalue()) * 100
            })

    def next(self):
        # 1. Date & Warmup
        if self.p.trading_start_date:
            if self.data.datetime.date(0) < self.p.trading_start_date: return
        if len(self) < self.p.sma_period: return
        
        # Record Equity
        if 'equity_curve' in st.session_state:
            st.session_state.equity_curve.append({
                'Date': self.data.datetime.date(0),
                'Equity': self.broker.getvalue()
            })

        # --- LOGIC ---
        
        if not self.position:
            # ENTRY LOGIC: "3 Down, 1 Up"
            
            # 1. Check Trend (Safety) - Price must be above 200 SMA
            is_uptrend = self.data.close[0] > self.sma200[0]
            
            # 2. Check "3 Consecutive Down Days" (Looking back from Yesterday)
            # Today is index [0], Yesterday is [-1], etc.
            day_1_down = self.data.close[-1] < self.data.close[-2] # Yesterday < Day Before
            day_2_down = self.data.close[-2] < self.data.close[-3]
            day_3_down = self.data.close[-3] < self.data.close[-4]
            three_red_days = day_1_down and day_2_down and day_3_down
            
            # 3. Check "Market Going Up" (Today is Green)
            # Close is higher than Open (Candle is Green) OR Close is higher than Yesterday (Price Up)
            # Using Close > Close[-1] is safer for trend confirmation
            today_is_green = self.data.close[0] > self.data.close[-1]
            
            # FIRE: If Uptrend AND 3 Red Days AND Today is Green -> BUY
            if is_uptrend and three_red_days and today_is_green:
                self.buy_all_in()
        
        else:
            # EXIT LOGIC: Keep RSI Exit (Sell into Strength)
            # Note: Stop Loss is handled by notify_order automatically
            if self.rsi[0] > self.p.rsi_exit:
                self.close()

    def buy_all_in(self):
        cash = self.broker.get_cash()
        size = int((cash * self.p.max_exposure) / self.data.close[0])
        if size > 0:
            self.buy(size=size)

# --- STREAMLIT UI ---
st.title("📉 3-Down Sniper: Confirmation Strategy")
st.markdown("""
**Logic:** Wait for the dust to settle.
* **Entry:** Buys ONLY after **3 Red Days** are followed by **1 Green Day**.
* **Trend Filter:** Only takes trades if Price is above the 200-Day SMA.
* **Exit:** Sells when RSI bounces above 70.
* **Safety:** Hard 5% Stop Loss.
""")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    asset_class = st.radio("Asset Class", ["Crypto", "US Stocks"])
    
    if asset_class == "Crypto":
        options = ["Use Custom Ticker", "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "WIF-USD", "PEPE-USD"]
    else:
        options = ["Use Custom Ticker", "NVDA", "TSLA", "AAPL", "MSFT", "AMD"]
        
    selected_option = st.selectbox("Select Asset", options, index=1)
    
    if selected_option == "Use Custom Ticker":
        ticker = st.text_input("Enter Ticker", value="BTC-USD").upper()
    else:
        ticker = selected_option
    
    st.divider()
    
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", datetime.date(2023, 1, 1))
    end_date = col2.date_input("End Date", datetime.date(2024, 12, 31))
    
    st.subheader("Strategy Settings")
    stop_loss_input = st.slider("Stop Loss (%)", 1, 15, 5)
    
    run_btn = st.button("🎯 Run 3-Down Bot", type="primary")

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
    with st.spinner(f" analyzing {ticker}..."):
        # --- INITIALIZE SESSION STATE ---
        st.session_state.equity_curve = [] 
        st.session_state.trades = [] 
        st.session_state.ticker = ticker 

        df = get_data(ticker, start_date, end_date)
        
        if not df.empty:
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            bh_df = df.loc[mask]
            
            if not bh_df.empty:
                # Buy & Hold Calc
                start_p = bh_df['Close'].iloc[0]
                end_p = bh_df['Close'].iloc[-1]
                bh_ret = ((end_p - start_p) / start_p) * 100
                bh_val = 100000 * (1 + bh_ret/100)
                
                # Run Backtest
                cerebro = bt.Cerebro()
                cerebro.addstrategy(ThreeDownStrategy, 
                                    stop_loss=stop_loss_input/100,
                                    trading_start_date=start_date)
                
                cerebro.adddata(bt.feeds.PandasData(dataname=df))
                cerebro.broker.setcash(100000)
                
                # Add 0.1% Commission
                cerebro.broker.setcommission(commission=0.001) 
                
                cerebro.run()
                
                final_val = cerebro.broker.getvalue()
                bot_ret = ((final_val / 100000) - 1) * 100
                
                # --- RESULTS ---
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Starting Capital", "$100,000")
                c2.metric("Final Value", f"${final_val:,.0f}", f"{bot_ret:.1f}%")
                c3.metric("Buy & Hold", f"${bh_val:,.0f}", f"{bh_ret:.1f}%")
                
                # --- CHARTING ---
                st.subheader("📈 Equity Curve")
                eq_df = pd.DataFrame(st.session_state.equity_curve)
                if not eq_df.empty:
                    eq_df = eq_df[eq_df['Date'] >= start_date]
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Scatter(x=eq_df['Date'], y=eq_df['Equity'], name="Bot Equity", line=dict(color="#00FF00", width=2)), secondary_y=False)
                    fig.add_trace(go.Scatter(x=bh_df.index, y=bh_df['Close'], name=f"{ticker} Price", line=dict(color="gray", dash='dot')), secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- TRADE LOG ---
                st.subheader("📋 Trade Ledger")
                trades_df = pd.DataFrame(st.session_state.trades)
                if not trades_df.empty:
                    st.dataframe(trades_df.style.map(
                        lambda x: f'background-color: {"#90EE90" if x > 0 else "#FFCCCB"}; color: black', 
                        subset=['PnL']
                    ), use_container_width=True)
                else:
                    st.info("No trades triggered. (Market might not have dropped 3 days in a row)")
            else:
                st.error("Not enough data.")
        else:
            st.error("Data download failed.")
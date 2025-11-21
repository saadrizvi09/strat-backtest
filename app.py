import streamlit as st
import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Sniper Bot Ultimate", layout="wide", page_icon="🦁")

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

# --- STRATEGY: AGGRESSIVE + STOP LOSS + COOL DOWN ---
class SniperStrategy(bt.Strategy):
    params = (
        ('sma_period', 200),      
        ('rsi_period', 2),        
        ('rsi_entry', 10),        
        ('rsi_exit', 70),         
        ('stop_loss', 0.05),      
        ('max_exposure', 0.95),
        ('cooldown_period', 5),   # 5 Day Ban after loss
        ('trading_start_date', None),
    )

    def __init__(self):
        self.sma200 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_period)
        self.rsi = SafeRSI(self.data.close, period=self.p.rsi_period)
        self.next_allowed_trade_bar = 0 # Track Cool Down

    def notify_trade(self, trade):
        if not trade.isclosed: return
        
        pnl = trade.pnlcomm
        entry_dt = bt.num2date(trade.dtopen)
        exit_dt = bt.num2date(trade.dtclose)
        
        # 1. FIX EXIT PRICE DISPLAY
        if len(trade.history) > 0:
            initial_size = trade.history[0].event.size
            price_diff = pnl / initial_size
            exit_price = trade.price + price_diff
        else:
            exit_price = trade.price

        # 2. ACTIVATE COOL DOWN IF LOSS
        if pnl < 0:
            self.next_allowed_trade_bar = len(self) + self.p.cooldown_period

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
        if self.p.trading_start_date and self.data.datetime.date(0) < self.p.trading_start_date: return
        if len(self) < self.p.sma_period: return
        
        if 'equity_curve' in st.session_state:
            st.session_state.equity_curve.append({
                'Date': self.data.datetime.date(0),
                'Equity': self.broker.getvalue()
            })

        # --- 1. CHECK COOL DOWN ---
        if len(self) < self.next_allowed_trade_bar:
            return 

        # --- 2. STRATEGY LOGIC ---
        if self.position:
            # STOP LOSS (Intraday Simulation)
            entry_price = self.position.price
            stop_price = entry_price * (1.0 - self.p.stop_loss)
            
            # Check if Low hit the stop
            if self.data.low[0] < stop_price:
                self.close(price=stop_price) # Force exit at stop price
                return 

            # TAKE PROFIT
            if self.rsi[0] > self.p.rsi_exit:
                self.close()

        else:
            # ENTRY (Aggressive)
            if self.data.close[0] > self.sma200[0] and self.rsi[0] < self.p.rsi_entry:
                self.buy_all_in()

    def buy_all_in(self):
        cash = self.broker.get_cash()
        size = int((cash * self.p.max_exposure) / self.data.close[0])
        if size > 0:
            self.buy(size=size)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    asset_class = st.radio("Asset Class", ["Crypto", "US Stocks"])
    
    if asset_class == "Crypto":
        majors = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "XRP-USD", "AVAX-USD"]
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
    start_date = col1.date_input("Start", datetime.date(2024, 1, 1))
    end_date = col2.date_input("End", datetime.date(2025, 12, 31))
    
    st.subheader("Strategy Settings")
    rsi_entry = st.slider("RSI Entry (<)", 2, 20, 10)
    
    run_btn = st.button("🦁 Run Ultimate Sniper", type="primary")

def get_data(ticker, start, end):
    warmup = start - datetime.timedelta(days=300)
    try:
        df = yf.download(ticker, start=warmup, end=end, interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df
    except: return pd.DataFrame()

if run_btn:
    with st.spinner(f"Hunting {ticker}..."):
        # --- INITIALIZE SESSION STATE ---
        st.session_state.equity_curve = [] 
        st.session_state.trades = [] 
        st.session_state.ticker = ticker  # <--- THIS WAS MISSING
        
        df = get_data(ticker, start_date, end_date)
        
        if not df.empty:
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            bh_df = df.loc[mask]
            if not bh_df.empty:
                bh_val = 100000 * (1 + ((bh_df['Close'].iloc[-1] - bh_df['Close'].iloc[0])/bh_df['Close'].iloc[0]))
                
                cerebro = bt.Cerebro()
                cerebro.addstrategy(SniperStrategy, 
                                    rsi_entry=rsi_entry,
                                    trading_start_date=start_date)
                
                cerebro.adddata(bt.feeds.PandasData(dataname=df))
                cerebro.broker.setcash(100000)
                
                # --- ADDED REALISTIC COMMISSION (0.1%) ---
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
                    fig.add_trace(go.Scatter(x=eq_df['Date'], y=eq_df['Equity'], name="Bot Equity", line=dict(color="#00FF00", width=2)), secondary_y=False)
                    fig.add_trace(go.Scatter(x=bh_df.index, y=bh_df['Close'], name="Price", line=dict(color="gray", dash='dot')), secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                trades_df = pd.DataFrame(st.session_state.trades)
                if not trades_df.empty:
                    st.dataframe(trades_df.style.map(
                        lambda x: f'background-color: {"#90EE90" if x > 0 else "#FFCCCB"}; color: black', 
                        subset=['PnL']
                    ), use_container_width=True)
                else:
                    st.info("No trades triggered.")
        else:
            st.error("Data download failed.")
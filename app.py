import streamlit as st
import backtrader as bt
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Saad Rizvi Gandphad strategy", layout="wide", page_icon="🚀")
st.markdown(
    """
    <style>
    .fixed-title {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #0E1117;  /* Dark background */
        color: white;
        text-align: center;
        padding-top: 15px;
        padding-bottom: 15px;
        font-size: 30px;
        font-weight: bold;
        z-index: 999999; /* Super high to sit on top of everything */
        border-bottom: 2px solid #ff4b4b; /* Red border to verify it's working */
    }
    /* Push main content down so it doesn't hide behind the title */
    .block-container {
        padding-top: 100px !important;
    }
    /* Hide the standard Streamlit top decoration bar if it gets in the way */
    header[data-testid="stHeader"] {
        z-index: 1;
    }
    </style>
    
    <div class="fixed-title">Saad Rizvi Gandphad Strategy</div>
    """,
    unsafe_allow_html=True
)
# --- 1. THE J/K MOMENTUM STRATEGY ---
class JKMomentumStrategy(bt.Strategy):
    """
    Replication of the J/K Momentum Strategy.
    1. Look back J periods to calculate returns.
    2. Rank assets (Winners).
    3. Hold for K periods before re-evaluating.
    """
    params = (
        ('J', 12),       # Formation Period (Lookback)
        ('K', 12),       # Holding Period (Rebalance Interval)
        ('top_n', 3),    # Number of assets to buy (The "Winners")
    )

    def __init__(self):
        self.inds = {}
        self.rebalance_counter = 0
        
        # Calculate Returns for every asset in the universe
        for d in self.datas:
            # ROC = Rate of Change (100 * (Price_now - Price_J_ago) / Price_J_ago)
            self.inds[d] = bt.indicators.RateOfChange(d.close, period=self.p.J)

    def next(self):
        # 1. Check if it is time to rebalance (Every K bars)
        self.rebalance_counter += 1
        if self.rebalance_counter < self.p.K:
            return
        
        # Reset counter
        self.rebalance_counter = 0

        # 2. Rank Assets
        # Get list of (data_feed, return_value)
        # Filter out assets that haven't been around long enough for J periods
        candidates = []
        for d, roc in self.inds.items():
            if len(d) > self.p.J and not pd.isna(roc[0]):
                candidates.append((d, roc[0]))

        # Sort by return (Highest to Lowest)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # 3. Identify Winners (Top N)
        winners = [x[0] for x in candidates[:self.p.top_n]]
        
        # 4. Execution Logic
        # Weight per asset (Simple Equal Weighting for stability)
        # If we pick top 3, each gets 33% of portfolio
        target_weight = 1.0 / self.p.top_n if self.p.top_n > 0 else 0

        # Rebalance current positions
        # a. Close positions that are no longer winners
        for d in self.datas:
            if d not in winners:
                self.order_target_percent(d, target=0.0)
        
        # b. Buy/Adjust positions for winners
        for d in winners:
            self.order_target_percent(d, target=target_weight)

        # Log for the UI
        if 'rebalance_log' in st.session_state:
             st.session_state.rebalance_log.append({
                'Date': self.datas[0].datetime.date(0),
                'Top Picks': [d._name for d in winners]
            })

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if 'trades' in st.session_state:
                st.session_state.trades.append({
                    'Date': bt.num2date(order.executed.dt).date(),
                    'Symbol': order.data._name,
                    'Action': 'BUY' if order.isbuy() else 'SELL',
                    'Price': order.executed.price,
                    'Size': order.executed.size,
                    'Value': order.executed.value
                })

# --- 2. HELPER FUNCTIONS ---
@st.cache_data
def get_universe_data(tickers, start, end):
    """Download data for multiple tickers and align them."""
    data_feed = {}
    # Add buffer for J period
    warmup_start = start - datetime.timedelta(days=150) 
    
    for t in tickers:
        try:
            df = yf.download(t, start=warmup_start, end=end, interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                data_feed[t] = df
        except Exception as e:
            print(f"Failed {t}: {e}")
            
    return data_feed

# --- 3. MAIN APP UI ---
with st.sidebar:
    st.header("Saad Rizvi strat")
    
    # Universe Selection
    universe_type = st.radio("Select Universe", ["Crypto Majors", "Tech Stocks", "Meme Coins"])
    
    if universe_type == "Crypto Majors":
        tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD", "XRP-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "LTC-USD"]
    elif universe_type == "Meme Coins":
        tickers = ["DOGE-USD", "SHIB-USD", "PEPE-USD", "WIF-USD", "BONK-USD", "FLOKI-USD"]
    else:
        tickers = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "GOOGL", "META", "AMZN", "NFLX", "INTC"]

    st.write(f"**Tracking {len(tickers)} Assets**")
    
    st.divider()
    
    # Strategy Parameters
    col1, col2 = st.columns(2)
    J_period = col1.number_input("J (Lookback)", min_value=5, max_value=365, value=30, help="Formation Period in Days")
    K_period = col2.number_input("K (Hold)", min_value=1, max_value=365, value=7, help="Holding Period in Days")
    
    top_n = st.slider("Select Top N Winners", 1, len(tickers), 3)
    
    st.divider()
    start_date = st.date_input("Start Date", datetime.date(2023, 1, 1))
    run_btn = st.button("🚀 Run Backtest", type="primary")

# --- 4. EXECUTION LOGIC ---
if run_btn:
    st.session_state.trades = []
    st.session_state.rebalance_log = []
    
    with st.status("Fetching Market Data...", expanded=True) as status:
        data_dict = get_universe_data(tickers, start_date, datetime.date.today())
        status.write("Data downloaded.")
        
        if not data_dict:
            st.error("No data found.")
            st.stop()

        status.write("Initializing Cerebro Engine...")
        cerebro = bt.Cerebro()
        
        # Add Strategy
        cerebro.addstrategy(JKMomentumStrategy, J=J_period, K=K_period, top_n=top_n)
        
        # Add Data Feeds
        for ticker, df in data_dict.items():
            data = bt.feeds.PandasData(dataname=df, name=ticker)
            cerebro.adddata(data)

        # Analyzer: Equity Curve
        cerebro.addobserver(bt.observers.Value)
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')

        # Capital
        start_cash = 100000
        cerebro.broker.setcash(start_cash)
        cerebro.broker.setcommission(commission=0.001) # 0.1% fee

        status.write("Running Simulation...")
        results = cerebro.run()
        strat = results[0]
        
        end_cash = cerebro.broker.getvalue()
        roi = ((end_cash - start_cash) / start_cash) * 100
        
        status.update(label="Backtest Complete!", state="complete", expanded=False)

    # --- 5. RESULTS DASHBOARD ---
    st.title(f"Momentum Results (J={J_period}, K={K_period})")
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Initial Capital", f"${start_cash:,.0f}")
    m2.metric("Final Equity", f"${end_cash:,.0f}")
    m3.metric("Total Return", f"{roi:.2f}%", delta=f"{roi:.2f}%")

    # Extract Equity Curve
    # Note: Backtrader returns are time-indexed
    returns_dict = strat.analyzers.returns.get_analysis()
    returns_df = pd.DataFrame(list(returns_dict.items()), columns=['Date', 'Return'])
    returns_df['Date'] = pd.to_datetime(returns_df['Date'])
    
    # Reconstruct cumulative equity
    returns_df['Cumulative'] = (1 + returns_df['Return']).cumprod() * start_cash

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns_df['Date'], y=returns_df['Cumulative'], name="Strategy Equity", line=dict(color="#00cc96", width=2)))
    fig.update_layout(title="Portfolio Value Over Time", xaxis_title="Date", yaxis_title="Value ($)", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Tabs for details
    tab1, tab2 = st.tabs(["📜 Trade Log", "🔄 Rebalance History"])
    
    with tab1:
        if st.session_state.trades:
            trade_df = pd.DataFrame(st.session_state.trades)
            st.dataframe(trade_df, use_container_width=True)
        else:
            st.info("No trades were executed.")
            
    with tab2:
        if st.session_state.rebalance_log:
            # Process log to show which coins were held during which periods
            reb_df = pd.DataFrame(st.session_state.rebalance_log)
            st.dataframe(reb_df, use_container_width=True)
        else:
            st.info("No rebalance events recorded.")
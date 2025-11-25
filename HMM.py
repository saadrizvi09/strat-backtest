import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- Config ---
st.set_page_config(page_title="HMM Strategy Backtester", layout="wide")

# --- Helper Functions ---

@st.cache_data
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_metrics(df, strategy_col='Strategy_Value', benchmark_col='Buy_Hold_Value'):
    """Calculates CAG, Sharpe, Drawdown, etc."""
    stats = {}
    
    for col, name in [(strategy_col, 'Strategy'), (benchmark_col, 'Buy & Hold')]:
        # Returns
        initial = df[col].iloc[0]
        final = df[col].iloc[-1]
        total_return = (final - initial) / initial
        
        # Daily Returns
        daily_ret = df[col].pct_change().dropna()
        
        # Sharpe (Annualized, assuming 252 trading days for crypto? 365 actually)
        # Using 365 for crypto
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() != 0 else 0
        
        # Max Drawdown
        rolling_max = df[col].cummax()
        drawdown = (df[col] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        stats[name] = {
            "Total Return": f"{total_return:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}"
        }
    
    return pd.DataFrame(stats)

def train_hmm_model(train_df, n_states):
    """Trains HMM on historical data (In-Sample)."""
    # Features: Log Returns and Volatility
    X_train = train_df[['Log_Returns', 'Volatility']].values * 100
    
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_train)
    
    # Sort states by Volatility (State 0 = Lowest Risk)
    hidden_states = model.predict(X_train)
    state_vol = []
    for i in range(n_states):
        avg_vol = X_train[hidden_states == i, 1].mean()
        state_vol.append((i, avg_vol))
    state_vol.sort(key=lambda x: x[1])
    
    # Create mapping: {Random_ID: Sorted_ID}
    mapping = {old: new for new, (old, _) in enumerate(state_vol)}
    
    return model, mapping

# --- Main Logic ---

st.title("📈 HMM Strategy Backtester")
st.markdown("""
**The Strategy:**
1.  **Driver:** EMA Crossover (Fast EMA > Slow EMA = Bullish).
2.  **Filter:** HMM Regime Detection (Block buys if Volatility Regime is High).
3.  **Validation:** Trains HMM on past data (In-Sample), tests on future data (Out-of-Sample).
""")

# Sidebar Inputs
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", "BTC-USD")
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*4))
    end_date = st.date_input("End Date", datetime.now())
    
    st.divider()
    split_pct = st.slider("Train/Test Split", 0.5, 0.9, 0.7)
    st.caption("Example: 0.7 means we train HMM on first 70% of data, backtest on last 30%.")
    
    st.divider()
    short_window = st.number_input("Fast EMA", 12)
    long_window = st.number_input("Slow EMA", 26)
    n_states = st.slider("HMM States", 2, 4, 3)

if st.button("Run Backtest"):
    df = fetch_data(ticker, start_date, end_date)
    
    if df is None or len(df) < 100:
        st.error("Not enough data to backtest.")
    else:
        # 1. Feature Engineering
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Returns'].rolling(window=10).std()
        df['EMA_Short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df['EMA_Long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
        df = df.dropna()
        
        # 2. Split Data (Prevent Look-Ahead Bias)
        split_idx = int(len(df) * split_pct)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        st.info(f"Training HMM on {len(train_df)} days. Backtesting on {len(test_df)} days.")
        
        # 3. Train HMM (The "Historian")
        hmm_model, state_map = train_hmm_model(train_df, n_states)
        
        # 4. Run Backtest Loop (Vectorized logic is hard with HMM state matching, so we use a fast loop)
        # First, get regimes for the TEST set using the trained model
        X_test = test_df[['Log_Returns', 'Volatility']].values * 100
        raw_states = hmm_model.predict(X_test)
        
        # Apply the mapping (Map Random ID -> Sorted ID)
        # Note: If a state in Test wasn't seen well in Train, mapping handles it by key
        test_df['Regime'] = [state_map.get(s, s) for s in raw_states]
        
        # 5. Apply Strategy Logic
        # 0 = Low Vol, (n_states-1) = High Vol
        high_vol_state = n_states - 1 
        
        # Vectorized Signal Calculation
        test_df['Signal'] = np.where(test_df['EMA_Short'] > test_df['EMA_Long'], 1, 0)
        
        # Apply HMM Filter: If Signal is 1 BUT Regime is High Vol, Signal = 0
        test_df['Filtered_Signal'] = np.where(
            (test_df['Signal'] == 1) & (test_df['Regime'] != high_vol_state), 
            1, 
            0
        )
        
        # 6. Calculate Returns
        # Strategy returns: If we held the asset yesterday (Filtered_Signal shifted), we get today's return
        test_df['Strategy_Returns'] = test_df['Filtered_Signal'].shift(1) * test_df['Log_Returns']
        
        # Buy & Hold returns
        test_df['Buy_Hold_Returns'] = test_df['Log_Returns']
        
        # Cumulative Returns (Equity Curve)
        test_df['Strategy_Value'] = (1 + test_df['Strategy_Returns']).cumprod()
        test_df['Buy_Hold_Value'] = (1 + test_df['Buy_Hold_Returns']).cumprod()
        
        # Handle NaN from shift
        test_df.dropna(inplace=True)
        
        # --- RESULTS ---
        
        # Metrics Table
        metrics_df = calculate_metrics(test_df)
        st.subheader("Performance Metrics (Out-of-Sample)")
        st.table(metrics_df)
        
        # Equity Curve Chart
        st.subheader("Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Buy_Hold_Value'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Strategy_Value'], name='HMM + EMA Strategy', line=dict(color='#00CC96', width=2)))
        
        # Add Markers for "Saved Crashes" (Where HMM blocked a buy during a crash)
        # We find days where Signal was 1 (EMA said buy) but Filtered_Signal was 0 (HMM said NO)
        saved_crashes = test_df[(test_df['Signal'] == 1) & (test_df['Filtered_Signal'] == 0)]
        if not saved_crashes.empty:
            fig.add_trace(go.Scatter(
                x=saved_crashes.index, 
                y=saved_crashes['Strategy_Value'], 
                mode='markers', 
                name='HMM Blocked Trade',
                marker=dict(color='orange', symbol='x-thin', size=6)
            ))
            
        fig.update_layout(title="Strategy vs. Benchmark", xaxis_title="Date", yaxis_title="Growth (1.0 = Breakeven)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown Chart
        st.subheader("Drawdown Analysis")
        # Calculate drawdowns
        strat_dd = (test_df['Strategy_Value'] - test_df['Strategy_Value'].cummax()) / test_df['Strategy_Value'].cummax()
        bench_dd = (test_df['Buy_Hold_Value'] - test_df['Buy_Hold_Value'].cummax()) / test_df['Buy_Hold_Value'].cummax()
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=test_df.index, y=bench_dd, name='Buy & Hold DD', line=dict(color='gray', width=1), fill='tozeroy'))
        fig_dd.add_trace(go.Scatter(x=test_df.index, y=strat_dd, name='Strategy DD', line=dict(color='red', width=1), fill='tozeroy'))
        fig_dd.update_layout(title="Drawdown Underwater Plot", xaxis_title="Date", yaxis_title="Drawdown %")
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Regime Analysis
        st.subheader("Regime Classification on Backtest Data")
        
        # Create a color map for the regimes
        regime_colors = {0: 'green', 1: 'blue', 2: 'red', 3: 'black'}
        colors = [regime_colors.get(r, 'gray') for r in test_df['Regime']]
        
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(
            x=test_df.index, 
            y=test_df['Close'],
            mode='markers+lines',
            marker=dict(color=colors, size=4),
            line=dict(color='lightgray', width=1),
            name='Price (Colored by Regime)'
        ))
        
        # Legend hack
        fig_reg.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='green'), name='Low Vol (Trade ON)'))
        fig_reg.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='red'), name='High Vol (Trade OFF)'))
        
        fig_reg.update_layout(title="Market Regimes during Backtest", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_reg, use_container_width=True)
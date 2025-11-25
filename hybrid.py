import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- Config ---
st.set_page_config(page_title="Hybrid HMM-SVR Strategy Backtester", layout="wide")

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
    
    for col, name in [(strategy_col, 'Hybrid Strategy'), (benchmark_col, 'Buy & Hold')]:
        # Returns
        initial = df[col].iloc[0]
        final = df[col].iloc[-1]
        total_return = (final - initial) / initial
        
        # Daily Returns
        daily_ret = df[col].pct_change().dropna()
        
        # Sharpe (Annualized, assuming 365 trading days for crypto)
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

def train_svr_model(train_df):
    """Trains SVR to predict next day's volatility."""
    # Features for SVR: Returns, Current Vol, Downside Vol, Regime
    feature_cols = ['Log_Returns', 'Volatility', 'Downside_Vol', 'Regime']
    target_col = 'Target_Next_Vol'
    
    X = train_df[feature_cols].values
    y = train_df[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SVR with RBF kernel
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
    model.fit(X_scaled, y)
    
    return model, scaler

# --- Main Logic ---

st.title("🧠 Hybrid HMM-SVR Strategy Backtester")
st.markdown("""
**The Hybrid Strategy:**
1.  **Driver:** EMA Crossover (Fast > Slow = Bullish).
2.  **Filter (HMM):** If Regime is "High Vol/Crash", **Block Trade** (Size = 0).
3.  **Sizing (SVR):** If Regime is Safe, adjust size based on predicted risk. 
    * *If SVR predicts higher risk -> Reduce Position Size.*
    * *If SVR predicts lower risk -> Increase Position Size.*
""")

# Sidebar Inputs
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", "BTC-USD")
    
    # Modified Date Logic: User selects Trading Period
    backtest_start = st.date_input("Backtest Start Date", datetime.now() - timedelta(days=365))
    backtest_end = st.date_input("Backtest End Date", datetime.now())
    
    st.caption("Note: Models will automatically train on the **4 years** of data prior to your selected Start Date.")
    
    st.divider()
    short_window = st.number_input("Fast EMA", 12)
    long_window = st.number_input("Slow EMA", 26)
    n_states = st.slider("HMM States", 2, 4, 3)

if st.button("Run Hybrid Backtest"):
    # Calculate the Training Start Date (4 Years before Backtest Start)
    train_start_date = pd.Timestamp(backtest_start) - pd.DateOffset(years=4)
    
    # Fetch ALL data (Training Period + Backtest Period)
    df = fetch_data(ticker, train_start_date, backtest_end)
    
    if df is None or len(df) < 200:
        st.error("Not enough data to backtest. Ensure the ticker existed 4 years prior to your start date.")
    else:
        # 1. Feature Engineering
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Returns'].rolling(window=10).std()
        
        # Downside Volatility (Leverage Effect Feature)
        df['Downside_Returns'] = df['Log_Returns'].apply(lambda x: x if x < 0 else 0)
        df['Downside_Vol'] = df['Downside_Returns'].rolling(window=10).std()
        
        # Strategy Indicators
        df['EMA_Short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df['EMA_Long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
        
        # Target for SVR (Next Day Volatility)
        df['Target_Next_Vol'] = df['Volatility'].shift(-1)
        
        df = df.dropna()
        
        # 2. Split Data based on Dates
        train_df = df[df.index < pd.Timestamp(backtest_start)].copy()
        test_df = df[df.index >= pd.Timestamp(backtest_start)].copy()
        
        if len(train_df) < 365:
            st.warning(f"Warning: Only {len(train_df)} days found for training. HMM performs best with >2 years of data.")
        
        if len(test_df) < 10:
             st.error("Not enough data for backtesting range.")
        else:
            st.info(f"Training on {len(train_df)} days ({train_df.index[0].date()} to {train_df.index[-1].date()}). Backtesting on {len(test_df)} days.")
            
            with st.spinner("Training HMM (Regime Detection)..."):
                hmm_model, state_map = train_hmm_model(train_df, n_states)
                
                # Predict Train Regimes (Needed for SVR training input)
                X_train_hmm = train_df[['Log_Returns', 'Volatility']].values * 100
                train_raw_states = hmm_model.predict(X_train_hmm)
                train_df['Regime'] = [state_map.get(s, s) for s in train_raw_states]
                
            with st.spinner("Training SVR (Volatility Forecasting)..."):
                svr_model, svr_scaler = train_svr_model(train_df)
                
            with st.spinner("Running Backtest Loop..."):
                # --- OUT OF SAMPLE BACKTEST ---
                
                # 1. Predict Regimes for Test Data
                X_test_hmm = test_df[['Log_Returns', 'Volatility']].values * 100
                test_raw_states = hmm_model.predict(X_test_hmm)
                test_df['Regime'] = [state_map.get(s, s) for s in test_raw_states]
                
                # 2. Predict Volatility for Test Data (Using SVR)
                X_test_svr = test_df[['Log_Returns', 'Volatility', 'Downside_Vol', 'Regime']].values
                X_test_svr_scaled = svr_scaler.transform(X_test_svr)
                test_df['Predicted_Vol'] = svr_model.predict(X_test_svr_scaled)
                
                # 3. Calculate Strategy Logic
                high_vol_state = n_states - 1
                
                # Base Signal (EMA)
                test_df['Signal'] = np.where(test_df['EMA_Short'] > test_df['EMA_Long'], 1, 0)
                
                # Calculate Baseline Risk (Average Volatility seen in Training)
                avg_train_vol = train_df['Volatility'].mean()
                
                # Calculate Position Size (The "Dimmer Switch")
                # Logic: Size = Average_Vol / Predicted_Vol
                # If Predicted > Average, Size < 1.0 (Reduce Risk)
                # If Predicted < Average, Size > 1.0 (Increase Risk) -> Capped at 1.0 for safety
                test_df['Risk_Ratio'] = test_df['Predicted_Vol'] / avg_train_vol
                test_df['Position_Size'] = (1.0 / test_df['Risk_Ratio']).clip(upper=1.0, lower=0.0)
                
                # Override: If HMM says CRASH, Size = 0
                test_df['Position_Size'] = np.where(
                    test_df['Regime'] == high_vol_state, 
                    0.0, 
                    test_df['Position_Size']
                )
                
                # Final Position: Signal * Size
                # We shift(1) because we calculate size today for tomorrow's return
                test_df['Final_Position'] = (test_df['Signal'] * test_df['Position_Size']).shift(1)
                
                # 4. Returns
                test_df['Strategy_Returns'] = test_df['Final_Position'] * test_df['Log_Returns']
                test_df['Buy_Hold_Returns'] = test_df['Log_Returns']
                
                # Cumulative
                test_df['Strategy_Value'] = (1 + test_df['Strategy_Returns']).cumprod()
                test_df['Buy_Hold_Value'] = (1 + test_df['Buy_Hold_Returns']).cumprod()
                test_df.dropna(inplace=True)
                
                # --- RESULTS ---
                
                metrics_df = calculate_metrics(test_df)
                st.subheader("Performance Metrics")
                st.table(metrics_df)
                
                # Charts
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Equity Curve")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Buy_Hold_Value'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
                    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Strategy_Value'], name='Hybrid Strategy', line=dict(color='#00CC96', width=2)))
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    st.subheader("Position Sizing (SVR Effect)")
                    st.caption("How SVR adjusted trade size over time (0.0 to 1.0)")
                    fig_size = px.area(test_df, x=test_df.index, y='Position_Size', title="Dynamic Exposure")
                    st.plotly_chart(fig_size, use_container_width=True)
                
                st.subheader("SVR Prediction Accuracy (Test Set)")
                fig_svr = go.Figure()
                # Show a slice to avoid clutter
                slice_df = test_df.iloc[-100:] 
                fig_svr.add_trace(go.Scatter(x=slice_df.index, y=slice_df['Target_Next_Vol'], name='Actual Volatility'))
                fig_svr.add_trace(go.Scatter(x=slice_df.index, y=slice_df['Predicted_Vol'], name='SVR Prediction', line=dict(dash='dot')))
                st.plotly_chart(fig_svr, use_container_width=True)
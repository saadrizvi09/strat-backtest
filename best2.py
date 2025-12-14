import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date

# --- Config ---
st.set_page_config(page_title="HMM-SVR Leverage Sniper", layout="wide")

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_data(ticker, start_date, end_date):
    """
    Robust data fetching with caching.
    """
    ticker = ticker.strip().upper()
    
    if isinstance(start_date, (datetime, pd.Timestamp)):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, (datetime, pd.Timestamp)):
        end_date = end_date.strftime('%Y-%m-%d')
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.dropna(how='all')
        
        if len(df) < 10:
            return None
            
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_metrics(df, strategy_col='Strategy_Value', benchmark_col='Buy_Hold_Value'):
    """Calculates CAG, Sharpe, Drawdown, etc."""
    stats = {}
    
    for col, name in [(strategy_col, 'Smart Leverage Strategy'), (benchmark_col, 'Buy & Hold')]:
        initial = df[col].iloc[0]
        final = df[col].iloc[-1]
        total_return = (final - initial) / initial
        
        daily_ret = df[col].pct_change().dropna()
        
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() != 0 else 0
        
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
    """Trains HMM on historical data and sorts states by volatility (0=Low, n=High)."""
    X_train = train_df[['Log_Returns', 'Volatility']].values * 100
    
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_train)
    
    hidden_states = model.predict(X_train)
    state_vol = []
    for i in range(n_states):
        avg_vol = X_train[hidden_states == i, 1].mean()
        state_vol.append((i, avg_vol))
    
    # Sort states: State 0 = Lowest Volatility (Safe), State N = Highest Volatility (Crash)
    state_vol.sort(key=lambda x: x[1])
    
    mapping = {old: new for new, (old, _) in enumerate(state_vol)}
    
    return model, mapping

def train_svr_model(train_df):
    """Trains SVR to predict next day's volatility."""
    feature_cols = ['Log_Returns', 'Volatility', 'Downside_Vol', 'Regime']
    target_col = 'Target_Next_Vol'
    
    X = train_df[feature_cols].values
    y = train_df[target_col].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
    model.fit(X_scaled, y)
    
    return model, scaler

def generate_trade_log(df):
    """Generates a log of trades including leverage used."""
    trades = []
    in_trade = False
    entry_date = None
    entry_price = 0
    trade_returns = []
    avg_leverage = []
    
    for date, row in df.iterrows():
        pos = row['Final_Position']
        close_price = row['Close']
        lev = row['Position_Size'] # Capture leverage used
        
        if pos > 0 and not in_trade:
            in_trade = True
            entry_date = date
            entry_price = close_price
            trade_returns = [row['Strategy_Returns']]
            avg_leverage = [lev]
            
        elif pos > 0 and in_trade:
            trade_returns.append(row['Strategy_Returns'])
            avg_leverage.append(lev)
            
        elif pos == 0 and in_trade:
            in_trade = False
            exit_date = date
            exit_price = close_price
            
            cum_trade_ret = np.prod([1 + r for r in trade_returns]) - 1
            mean_lev = np.mean(avg_leverage)
            
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Duration': len(trade_returns),
                'Avg Leverage': f"{mean_lev:.1f}x",
                'Trade PnL': cum_trade_ret
            })
            trade_returns = []
            avg_leverage = []

    if in_trade:
        cum_trade_ret = np.prod([1 + r for r in trade_returns]) - 1
        mean_lev = np.mean(avg_leverage)
        trades.append({
            'Entry Date': entry_date,
            'Exit Date': df.index[-1],
            'Entry Price': entry_price,
            'Exit Price': df.iloc[-1]['Close'],
            'Duration': len(trade_returns),
            'Avg Leverage': f"{mean_lev:.1f}x",
            'Trade PnL': cum_trade_ret
        })

    return pd.DataFrame(trades)

# --- Main Logic ---

st.title("⚡ HMM-SVR  Leverage Backtester")
st.markdown("""
**The "Strict Rules" Strategy:**
1.  **Baseline:** Buy when Fast EMA > Slow EMA.
2.  **Safety (HMM):** If Regime = **High Volatility (Crash)** -> **Exit (0x)**.
3.  **Leverage Boost (SVR + HMM):** * IF Regime is **Lowest Volatility (State 0)**
    * AND SVR predicts volatility **< 50% of average** (Risk Ratio < 0.5)
    * THEN **Leverage = 3x**.
""")

# Sidebar Inputs
with st.sidebar:
    st.header("Settings")
    
    ticker = st.selectbox(
        "Ticker", 
        ["BNB-USD", "ETH-USD", "SOL-USD", "BTC-USD"],
        key="ticker_select" 
    )
    
    backtest_start = st.date_input(
        "Backtest Start Date", 
        date(2022, 1, 1),
        key="start_date"
    )

    backtest_end = st.date_input(
        "Backtest End Date", 
        datetime.now(),
        key="end_date"
    )
    
    st.divider()
    
    st.subheader("Leverage Rules")
    leverage_mult = st.number_input("Boost Leverage (Certainty Multiplier)", value=3.0, step=0.5)
    risk_threshold = st.slider("Certainty Threshold (Risk Ratio < X)", 0.1, 1.0, 0.5, help="Lower = Stricter. Only boost leverage when predicted risk is extremely low.")

if st.button("Run Leverage Backtest"):
    train_start_date = pd.Timestamp(backtest_start) - pd.DateOffset(years=4)
    
    df = fetch_data(ticker, train_start_date, backtest_end)
    
    if df is None or len(df) < 200:
        st.error(f"Not enough data found for {ticker}.")
    else:
        # 1. Feature Engineering
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Returns'].rolling(window=10).std()
        
        df['Downside_Returns'] = df['Log_Returns'].apply(lambda x: x if x < 0 else 0)
        df['Downside_Vol'] = df['Downside_Returns'].rolling(window=10).std()
        
        # 12/26 EMA standard
        df['EMA_Short'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_Long'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        df['Target_Next_Vol'] = df['Volatility'].shift(-1)
        
        df = df.dropna()
        
        # 2. Split Data
        train_df = df[df.index < pd.Timestamp(backtest_start)].copy()
        test_df = df[df.index >= pd.Timestamp(backtest_start)].copy()
        
        if len(train_df) < 365:
            st.warning(f"Warning: Only {len(train_df)} days found for training.")
        
        if len(test_df) < 10:
             st.error("Not enough data for backtesting range.")
        else:
            n_states = 3 # Fixed 3 states: Low, Neutral, High
            
            with st.spinner("Training AI Models (HMM & SVR)..."):
                # Train HMM
                hmm_model, state_map = train_hmm_model(train_df, n_states)
                
                # Get HMM Regimes for Train set (needed for SVR training)
                X_train_hmm = train_df[['Log_Returns', 'Volatility']].values * 100
                train_raw_states = hmm_model.predict(X_train_hmm)
                train_df['Regime'] = [state_map.get(s, s) for s in train_raw_states]
                
                # Train SVR
                svr_model, svr_scaler = train_svr_model(train_df)
                
                # --- OUT OF SAMPLE BACKTEST ---
                
                # Predict Regimes
                X_test_hmm = test_df[['Log_Returns', 'Volatility']].values * 100
                test_raw_states = hmm_model.predict(X_test_hmm)
                test_df['Regime'] = [state_map.get(s, s) for s in test_raw_states]
                
                # Predict Next Day Volatility
                X_test_svr = test_df[['Log_Returns', 'Volatility', 'Downside_Vol', 'Regime']].values
                X_test_svr_scaled = svr_scaler.transform(X_test_svr)
                test_df['Predicted_Vol'] = svr_model.predict(X_test_svr_scaled)
                
                # --- STRICT LEVERAGE LOGIC ---
                
                # 1. Base Signal (Trend)
                test_df['Signal'] = np.where(test_df['EMA_Short'] > test_df['EMA_Long'], 1, 0)
                
                # 2. Calculate Confidence
                avg_train_vol = train_df['Volatility'].mean()
                test_df['Risk_Ratio'] = test_df['Predicted_Vol'] / avg_train_vol
                
                # 3. Apply "The Rules"
                
                # Rule A: Default Size
                test_df['Position_Size'] = 1.0
                
                # Rule B: The "Certainty" Boost
                # If Regime is lowest volatility (State 0) AND Risk Ratio is low (< threshold)
                # Then apply leverage
                condition_safe_regime = (test_df['Regime'] == 0)
                condition_low_risk_prediction = (test_df['Risk_Ratio'] < risk_threshold)
                
                test_df['Position_Size'] = np.where(
                    condition_safe_regime & condition_low_risk_prediction,
                    leverage_mult, # User selected leverage (e.g., 2.0x)
                    test_df['Position_Size']
                )
                
                # Rule C: The "Danger" Cut
                # If Regime is Highest Volatility (State n-1) -> Go to 0
                condition_crash_regime = (test_df['Regime'] == (n_states - 1))
                
                test_df['Position_Size'] = np.where(
                    condition_crash_regime,
                    0.0,
                    test_df['Position_Size']
                )
                
                # Final Position Calculation
                # Shift by 1 because we act on Today's close for Tomorrow's return
                test_df['Final_Position'] = (test_df['Signal'] * test_df['Position_Size']).shift(1)
                
                # Returns Calculation
                test_df['Simple_Returns'] = test_df['Close'].pct_change()
                test_df['Strategy_Returns'] = test_df['Final_Position'] * test_df['Simple_Returns']
                
                # --- METRICS & VISUALS ---
                
                test_df['Strategy_Value'] = (1 + test_df['Strategy_Returns'].fillna(0)).cumprod()
                test_df['Buy_Hold_Value'] = (1 + test_df['Simple_Returns'].fillna(0)).cumprod()
                test_df.dropna(inplace=True)
                
                metrics_df = calculate_metrics(test_df)
                
                st.subheader("Performance vs Benchmark")
                st.table(metrics_df)
                
                # Plot 1: Equity Curve
                st.subheader("Equity Curve")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Buy_Hold_Value'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
                fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Strategy_Value'], name='Smart Leverage Strategy', line=dict(color='#00CC96', width=2)))
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot 2: Leverage Deployment
                st.subheader("Leverage Deployment (0x, 1x, or 2x)")
                st.caption("Notice how it shifts to 2x (Green Fill) only during smooth uptrends.")
                
                fig_lev = go.Figure()
                fig_lev.add_trace(go.Scatter(
                    x=test_df.index, 
                    y=test_df['Position_Size'], 
                    mode='lines',
                    fill='tozeroy',
                    name='Leverage Used',
                    line=dict(color='#636EFA')
                ))
                st.plotly_chart(fig_lev, use_container_width=True)
                
                # Trade Log
                st.divider()
                trade_log = generate_trade_log(test_df)
                st.subheader("📝 Leverage Trade Log")
                if not trade_log.empty:
                    # Formatting
                    display_log = trade_log.copy()
                    display_log['Entry Date'] = display_log['Entry Date'].dt.date
                    display_log['Exit Date'] = display_log['Exit Date'].dt.date
                    display_log['Trade PnL'] = display_log['Trade PnL'].map('{:.2%}'.format)
                    display_log['Entry Price'] = display_log['Entry Price'].map('{:.2f}'.format)
                    display_log['Exit Price'] = display_log['Exit Price'].map('{:.2f}'.format)
                    st.dataframe(display_log, use_container_width=True)
                else:
                    st.write("No trades generated.")
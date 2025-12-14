import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, date

# --- Config ---
st.set_page_config(page_title="HMM-SVR Honest Leverage Sniper", layout="wide")

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_data(ticker, start_date, end_date):
    ticker = ticker.strip().upper()
    if isinstance(start_date, (datetime, pd.Timestamp)):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, (datetime, pd.Timestamp)):
        end_date = end_date.strftime('%Y-%m-%d')
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how='all')
        if len(df) < 10: return None
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def calculate_metrics(df, strategy_col='Strategy_Value', benchmark_col='Buy_Hold_Value'):
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
    X_train = train_df[['Log_Returns', 'Volatility']].values * 100
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_train)
    
    hidden_states = model.predict(X_train)
    state_vol = []
    for i in range(n_states):
        avg_vol = X_train[hidden_states == i, 1].mean()
        state_vol.append((i, avg_vol))
    
    state_vol.sort(key=lambda x: x[1])
    mapping = {old: new for new, (old, _) in enumerate(state_vol)}
    return model, mapping

def train_svr_model(train_df):
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
    trades = []
    in_trade = False
    entry_date = None
    entry_price = 0
    trade_returns = []
    avg_leverage = []
    
    for date, row in df.iterrows():
        pos = row['Final_Position']
        close_price = row['Close']
        lev = row['Position_Size']
        
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
                'Entry Date': entry_date, 'Exit Date': exit_date,
                'Entry Price': entry_price, 'Exit Price': exit_price,
                'Duration': len(trade_returns), 'Avg Leverage': f"{mean_lev:.1f}x",
                'Trade PnL': cum_trade_ret
            })
            trade_returns = []
            avg_leverage = []

    if in_trade:
        cum_trade_ret = np.prod([1 + r for r in trade_returns]) - 1
        mean_lev = np.mean(avg_leverage)
        trades.append({
            'Entry Date': entry_date, 'Exit Date': df.index[-1],
            'Entry Price': entry_price, 'Exit Price': df.iloc[-1]['Close'],
            'Duration': len(trade_returns), 'Avg Leverage': f"{mean_lev:.1f}x",
            'Trade PnL': cum_trade_ret
        })
    return pd.DataFrame(trades)

# --- Main Logic ---

st.title("⚡ HMM-SVR Honest Leverage Backtester")
st.markdown("""
**The "Strict Rules" Strategy (No Lookahead Bias):**
1. **Baseline:** Buy when Fast EMA > Slow EMA.
2. **Safety (HMM):** Calculates market regime using ONLY past data.
3. **Leverage Boost:** Uses SVR to predict *tomorrow's* volatility based on *today's* data.
**Timing:** Uses End-of-Day (EOD) data to make decisions for the next trading day.
""")

with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Ticker", ["BNB-USD", "ETH-USD", "SOL-USD", "BTC-USD"])
    backtest_start = st.date_input("Backtest Start Date", date(2022, 1, 1))
    backtest_end = st.date_input("Backtest End Date", datetime.now())
    st.divider()
    st.subheader("Leverage Rules")
    leverage_mult = st.number_input("Boost Leverage", value=3.0, step=0.5)
    risk_threshold = st.slider("Certainty Threshold", 0.1, 1.0, 0.5)

if st.button("Run Honest Backtest"):
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
        df['Target_Next_Vol'] = df['Volatility'].shift(-1)
        df = df.dropna()
        
        # 2. Split Data
        train_df = df[df.index < pd.Timestamp(backtest_start)].copy()
        test_df = df[df.index >= pd.Timestamp(backtest_start)].copy()
        
        if len(train_df) < 365 or len(test_df) < 10:
            st.error("Data split error. Adjust dates.")
        else:
            n_states = 3
            
            with st.spinner("1. Training Models on History..."):
                # Train HMM on Past Data
                hmm_model, state_map = train_hmm_model(train_df, n_states)
                
                # Get Regimes for Train set to train SVR
                X_train_hmm = train_df[['Log_Returns', 'Volatility']].values * 100
                train_raw_states = hmm_model.predict(X_train_hmm)
                train_df['Regime'] = [state_map.get(s, s) for s in train_raw_states]
                
                # Train SVR
                svr_model, svr_scaler = train_svr_model(train_df)
            
            # --- HONEST WALK-FORWARD BACKTEST ---
            
            st.info("2. Running Walk-Forward Simulation (Step-by-Step)... This simulates real-time trading.")
            progress_bar = st.progress(0)
            
            # Prepare lists for storing honest predictions
            honest_regimes = []
            honest_predicted_vols = []
            
            # Concatenate for sliding window access
            all_data = pd.concat([train_df, test_df])
            start_idx = len(train_df)
            total_steps = len(test_df)
            
            # We use a fixed lookback window for HMM inference to keep it fast enough
            # Looking back 252 days (1 year) is usually sufficient for regime detection
            lookback_window = 252 
            
            for i in range(total_steps):
                # Update UI
                if i % 10 == 0: progress_bar.progress((i + 1) / total_steps)
                
                # Define the window: From (Now - Lookback) to Now
                curr_pointer = start_idx + i
                window_start = max(0, curr_pointer - lookback_window)
                
                # Slice data strictly up to the current day 'i'
                # We include 'i' because we are making a decision at Close of day 'i' for the next day
                history_slice = all_data.iloc[window_start : curr_pointer + 1]  # Remove the +1
                
                # --- A. Honest Regime Detection ---
                # HMM determines the path of states that best fits this specific history
                X_slice = history_slice[['Log_Returns', 'Volatility']].values * 100
                
                try:
                    # Predict sequence
                    hidden_states_slice = hmm_model.predict(X_slice)
                    # We only care about the LAST state (the state of "Today")
                    current_state_raw = hidden_states_slice[-1]
                    current_state = state_map.get(current_state_raw, current_state_raw)
                except:
                    current_state = 1 # Fallback to Neutral if error
                
                honest_regimes.append(current_state)
                
                # --- B. Honest Volatility Prediction ---
                # Prepare single row input for SVR: [Log_Ret, Vol, Down_Vol, Regime]
                # Note: We use the 'current_state' we just calculated
                row = test_df.iloc[i]
                svr_features = np.array([[
                    row['Log_Returns'], 
                    row['Volatility'], 
                    row['Downside_Vol'], 
                    current_state
                ]])
                
                # Scale and Predict
                svr_feat_scaled = svr_scaler.transform(svr_features)
                pred_vol = svr_model.predict(svr_feat_scaled)[0]
                honest_predicted_vols.append(pred_vol)
                
                # --- Fix 1: Calculate EMAs properly in walk-forward ---
                # Calculate EMAs using only the history up to current day
                test_df.loc[test_df.index[i], 'EMA_Short'] = history_slice['Close'].ewm(span=12).mean().iloc[-1]
                test_df.loc[test_df.index[i], 'EMA_Long'] = history_slice['Close'].ewm(span=26).mean().iloc[-1]
            
            # Assign the honest predictions back to dataframe
            test_df['Regime'] = honest_regimes
            test_df['Predicted_Vol'] = honest_predicted_vols
            
            progress_bar.empty()
            
            # --- STRATEGY LOGIC (Same as before) ---
            
            test_df['Signal'] = np.where(test_df['EMA_Short'] > test_df['EMA_Long'], 1, 0)
            avg_train_vol = train_df['Volatility'].mean()
            test_df['Risk_Ratio'] = test_df['Predicted_Vol'] / avg_train_vol
            
            test_df['Position_Size'] = 1.0
            
            # Logic
            cond_safe = (test_df['Regime'] == 0)
            cond_low_risk = (test_df['Risk_Ratio'] < risk_threshold)
            cond_crash = (test_df['Regime'] == (n_states - 1))
            
            # Boost
            test_df['Position_Size'] = np.where(cond_safe & cond_low_risk, leverage_mult, test_df['Position_Size'])
            # Cut
            test_df['Position_Size'] = np.where(cond_crash, 0.0, test_df['Position_Size'])
            
            # Calculate Returns
            test_df['Final_Position'] = (test_df['Signal'] * test_df['Position_Size']).shift(1)
            test_df['Simple_Returns'] = test_df['Close'].pct_change()
            test_df['Strategy_Returns'] = test_df['Final_Position'] * test_df['Simple_Returns']
            
            # Metrics & Plots
            test_df['Strategy_Value'] = (1 + test_df['Strategy_Returns'].fillna(0)).cumprod()
            test_df['Buy_Hold_Value'] = (1 + test_df['Simple_Returns'].fillna(0)).cumprod()
            test_df.dropna(inplace=True)
            
            metrics_df = calculate_metrics(test_df)
            st.subheader("Performance vs Benchmark")
            st.table(metrics_df)
            
            st.subheader("Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Buy_Hold_Value'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Strategy_Value'], name='Smart Leverage', line=dict(color='#00CC96', width=2)))
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Leverage Deployment")
            fig_lev = go.Figure()
            fig_lev.add_trace(go.Scatter(x=test_df.index, y=test_df['Position_Size'], mode='lines', fill='tozeroy', name='Lev', line=dict(color='#636EFA')))
            st.plotly_chart(fig_lev, use_container_width=True)
            
            trade_log = generate_trade_log(test_df)
            st.subheader("📝 Trade Log")
            if not trade_log.empty:
                display_log = trade_log.copy()
                display_log['Trade PnL'] = display_log['Trade PnL'].map('{:.2%}'.format)
                st.dataframe(display_log, use_container_width=True)
            else:
                st.write("No trades generated.")
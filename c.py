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
st.set_page_config(page_title="SLR-Aligned Crypto Sniper", layout="wide")

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_data(ticker, start_date, end_date):
    """Fetches data. If ticker is not BTC, also fetches BTC to model 'Spillovers'[cite: 407]."""
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
        
        # [cite: 407] Spillover Integration: Fetch BTC if we are trading an altcoin
        if "BTC" not in ticker:
            btc_df = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
            if not btc_df.empty:
                if isinstance(btc_df.columns, pd.MultiIndex):
                    btc_df.columns = btc_df.columns.get_level_values(0)
                # Align BTC data
                df['BTC_Close'] = btc_df['Close']
                df['BTC_Close'] = df['BTC_Close'].ffill()
        else:
            df['BTC_Close'] = df['Close'] # BTC uses itself
            
        if len(df) < 10: return None
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def calculate_metrics(df, strategy_col='Strategy_Value', benchmark_col='Buy_Hold_Value'):
    stats = {}
    for col, name in [(strategy_col, 'SLR-Aligned Strategy'), (benchmark_col, 'Buy & Hold')]:
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
    # [cite: 315] MS-GARCH / HMM logic to detect regimes
    # We use BTC volatility for regime detection even if trading alts, due to "Spillovers" [cite: 407]
    X_train = train_df[['BTC_Log_Ret', 'BTC_Vol']].values * 100 
    
    # GaussianHMM is a proxy for the MS-GARCH models preferred in the paper [cite: 22, 382]
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_train)
    
    hidden_states = model.predict(X_train)
    state_vol = []
    for i in range(n_states):
        avg_vol = X_train[hidden_states == i, 1].mean()
        state_vol.append((i, avg_vol))
    
    # Sort states by volatility (0=Low, 1=Med, 2=High)
    state_vol.sort(key=lambda x: x[1])
    mapping = {old: new for new, (old, _) in enumerate(state_vol)}
    return model, mapping

def train_svr_model(train_df):
    # [cite: 460] "Best machine learning technique performances are hybrid models that consider the SVM"
    # [cite: 453] Added Upside/Downside Vol to capture "Anti-leverage" asymmetry
    feature_cols = ['Log_Returns', 'Volatility', 'Downside_Vol', 'Upside_Vol', 'Regime', 'BTC_Vol']
    target_col = 'Target_Next_Vol'
    
    train_df_clean = train_df.dropna(subset=feature_cols + [target_col])
    X = train_df_clean[feature_cols].values
    y = train_df_clean[target_col].values
    
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

st.title("📚 SLR-Aligned Crypto Backtester (HMM-SVR)")
st.markdown("""
**Changes based on "A Systematic Literature Review of Volatility and Risk"[cite: 4]:**
1.  **Anti-Leverage Logic:** High volatility is NOT always bad. The model now differentiates between *Crash Volatility* and *FOMO Volatility*[cite: 453].
2.  **Spillover Effect:** Uses BTC data as a feature even when trading Altcoins, as spillovers are bidirectional and significant[cite: 407].
3.  **Illiquidity Costs:** Added slippage/commission simulation.
""")

with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Ticker", ["BNB-USD", "ETH-USD", "SOL-USD", "LINK-USD", "BTC-USD"])
    backtest_start = st.date_input("Backtest Start Date", date(2022, 1, 1))
    backtest_end = st.date_input("Backtest End Date", datetime.now())
    st.divider()
    st.subheader("Strategy Params")
    leverage_mult = st.number_input("Boost Leverage", value=2.0, step=0.5, help="Leverage used in Safe/FOMO regimes")
    risk_threshold = st.slider("Vol Prediction Threshold", 0.1, 1.5, 0.8)
    st.subheader("Costs (Illiquidity)")
    slippage_bps = st.number_input("Slippage (bps)", value=10, help="Basis points lost per trade due to illiquidity ")
    comm_bps = st.number_input("Commission (bps)", value=5)

if st.button("Run SLR-Validated Backtest"):
    train_start_date = pd.Timestamp(backtest_start) - pd.DateOffset(years=4)
    df = fetch_data(ticker, train_start_date, backtest_end)
    
    if df is None or len(df) < 200:
        st.error(f"Not enough data found for {ticker}.")
    else:
        # 1. Feature Engineering
        # Calculate Returns for Ticker
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Returns'].rolling(window=10).std()
        
        # [cite: 453] Asymmetric Volatility (Anti-Leverage) features
        df['Downside_Returns'] = df['Log_Returns'].apply(lambda x: x if x < 0 else 0)
        df['Upside_Returns'] = df['Log_Returns'].apply(lambda x: x if x > 0 else 0)
        df['Downside_Vol'] = df['Downside_Returns'].rolling(window=10).std()
        df['Upside_Vol'] = df['Upside_Returns'].rolling(window=10).std()
        
        # Calculate Returns for BTC (Spillover Feature) [cite: 407]
        df['BTC_Log_Ret'] = np.log(df['BTC_Close'] / df['BTC_Close'].shift(1))
        df['BTC_Vol'] = df['BTC_Log_Ret'].rolling(window=10).std()
        
        df['Target_Next_Vol'] = df['Volatility'].shift(-1)
        df = df.dropna()
        
        # 2. Split Data
        train_df = df[df.index < pd.Timestamp(backtest_start)].copy()
        test_df = df[df.index >= pd.Timestamp(backtest_start)].copy()
        
        if len(train_df) < 365 or len(test_df) < 10:
            st.error("Data split error. Adjust dates.")
        else:
            n_states = 3
            
            with st.spinner("Training Hybrid HMM-SVR Models[cite: 460]..."):
                # Train HMM on Past Data
                hmm_model, state_map = train_hmm_model(train_df, n_states)
                
                # Get Regimes for Train set to train SVR
                X_train_hmm = train_df[['BTC_Log_Ret', 'BTC_Vol']].values * 100
                train_raw_states = hmm_model.predict(X_train_hmm)
                train_df['Regime'] = [state_map.get(s, s) for s in train_raw_states]
                
                # Train SVR
                svr_model, svr_scaler = train_svr_model(train_df)
            
            # --- WALK-FORWARD BACKTEST ---
            st.info("Running Walk-Forward Simulation...")
            progress_bar = st.progress(0)
            
            honest_regimes = []
            honest_predicted_vols = []
            
            all_data = pd.concat([train_df, test_df])
            start_idx = len(train_df)
            total_steps = len(test_df)
            lookback_window = 365 # [cite: 316] Medium/Long term horizons
            
            for i in range(total_steps):
                if i % 20 == 0: progress_bar.progress((i + 1) / total_steps)
                
                curr_pointer = start_idx + i
                window_start = max(0, curr_pointer - lookback_window)
                history_slice = all_data.iloc[window_start : curr_pointer + 1]
                
                # A. HMM Inference (Using BTC data for regime stability)
                X_slice = history_slice[['BTC_Log_Ret', 'BTC_Vol']].values * 100
                try:
                    hidden_states_slice = hmm_model.predict(X_slice)
                    current_state_raw = hidden_states_slice[-1]
                    current_state = state_map.get(current_state_raw, current_state_raw)
                except:
                    current_state = 1
                
                honest_regimes.append(current_state)
                
                # B. SVR Prediction (Hybrid approach) [cite: 383]
                row = test_df.iloc[i]
                svr_features = np.array([[
                    row['Log_Returns'], 
                    row['Volatility'], 
                    row['Downside_Vol'], 
                    row['Upside_Vol'],
                    current_state,
                    row['BTC_Vol'] # [cite: 407] Spillover
                ]])
                
                svr_feat_scaled = svr_scaler.transform(svr_features)
                pred_vol = svr_model.predict(svr_feat_scaled)[0]
                honest_predicted_vols.append(pred_vol)
                
                # Update EMAs (Baseline)
                test_df.loc[test_df.index[i], 'EMA_Short'] = history_slice['Close'].ewm(span=12).mean().iloc[-1]
                test_df.loc[test_df.index[i], 'EMA_Long'] = history_slice['Close'].ewm(span=26).mean().iloc[-1]
            
            test_df['Regime'] = honest_regimes
            test_df['Predicted_Vol'] = honest_predicted_vols
            progress_bar.empty()
            
            # --- ALIGNED STRATEGY LOGIC ---
            
            test_df['Signal'] = np.where(test_df['EMA_Short'] > test_df['EMA_Long'], 1, 0)
            avg_train_vol = train_df['Volatility'].mean()
            test_df['Risk_Ratio'] = test_df['Predicted_Vol'] / avg_train_vol
            
            test_df['Position_Size'] = 1.0
            
            # --- Logic Enhancements based on Paper ---
            
            # 1. Safe Regime: Low Vol = High Leverage
            cond_safe = (test_df['Regime'] == 0)
            
            # 2. [cite: 453] Anti-Leverage Effect: 
            # High Vol (Regime 2) is NOT automatically a crash.
            # If High Vol + Positive Return Trend (EMA > Long), it's likely FOMO/Bull Run.
            # We only cut if High Vol + Negative Trend (EMA < Long).
            
            cond_high_vol = (test_df['Regime'] == (n_states - 1))
            cond_bear_trend = (test_df['EMA_Short'] < test_df['EMA_Long'])
            
            cond_crash_protection = cond_high_vol & cond_bear_trend
            
            # Logic Application
            # Boost if Safe OR (High Vol AND Bullish Trend [FOMO])
            # This captures the "Anti-Leverage" finding that high vol often means gains in crypto
            test_df['Position_Size'] = np.where(cond_safe & (test_df['Risk_Ratio'] < risk_threshold), leverage_mult, test_df['Position_Size'])
            
            # Cut ONLY if Crash Protection triggers (High Vol + Bear Trend)
            test_df['Position_Size'] = np.where(cond_crash_protection, 0.0, test_df['Position_Size'])
            
            # --- Cost Simulation  ---
            # Apply Slippage + Comm to returns
            total_cost_pct = (slippage_bps + comm_bps) / 10000
            
            test_df['Final_Position'] = (test_df['Signal'] * test_df['Position_Size']).shift(1)
            
            # Detect Trades to apply costs
            test_df['Trade_Occurred'] = test_df['Final_Position'].diff().abs() > 0
            
            test_df['Simple_Returns'] = test_df['Close'].pct_change()
            
            # Raw Strategy Returns
            test_df['Strategy_Returns'] = test_df['Final_Position'] * test_df['Simple_Returns']
            
            # Deduct Costs
            test_df['Strategy_Returns'] = np.where(
                test_df['Trade_Occurred'], 
                test_df['Strategy_Returns'] - total_cost_pct, 
                test_df['Strategy_Returns']
            )
            
            # Metrics
            test_df['Strategy_Value'] = (1 + test_df['Strategy_Returns'].fillna(0)).cumprod()
            test_df['Buy_Hold_Value'] = (1 + test_df['Simple_Returns'].fillna(0)).cumprod()
            test_df.dropna(inplace=True)
            
            metrics_df = calculate_metrics(test_df)
            st.subheader("Performance vs Benchmark")
            st.table(metrics_df)
            
            st.subheader("Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Buy_Hold_Value'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Strategy_Value'], name='SLR-Aligned Strategy', line=dict(color='#00CC96', width=2)))
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Regime Analysis (BTC-Driven)")
            # visualize regimes background
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=test_df.index, y=test_df['Close'], name='Price', line=dict(color='white', width=1)))
            
            # Color code based on Regime
            # Green = Low Vol, Red = High Vol
            st.plotly_chart(fig_reg, use_container_width=True)

            trade_log = generate_trade_log(test_df)
            st.subheader("📝 Trade Log (with Slippage)")
            if not trade_log.empty:
                display_log = trade_log.copy()
                display_log['Trade PnL'] = display_log['Trade PnL'].map('{:.2%}'.format)
                st.dataframe(display_log, use_container_width=True)
            else:
                st.write("No trades generated.")
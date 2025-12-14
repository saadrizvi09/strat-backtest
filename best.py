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
st.set_page_config(page_title="Leveraged HMM-SVR Quant Trader", layout="wide")

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def fetch_data(ticker, start_date, end_date):
    """
    Robust data fetching with caching, error handling, and string conversion.
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
    
    for col, name in [(strategy_col, 'Hybrid Strategy'), (benchmark_col, 'Buy & Hold')]:
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
    """Trains HMM on historical data (In-Sample)."""
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
    """
    Scans the backtest dataframe to identify individual trade cycles.
    """
    trades = []
    in_trade = False
    entry_date = None
    entry_price = 0
    trade_returns = []
    
    for date, row in df.iterrows():
        pos = row['Final_Position']
        close_price = row['Close']
        
        # Check for Entry (Position goes from 0 to > 0)
        if pos > 0 and not in_trade:
            in_trade = True
            entry_date = date
            entry_price = close_price 
            trade_returns = [row['Strategy_Returns']]
            
        # Check for adjustments while in trade
        elif pos > 0 and in_trade:
            trade_returns.append(row['Strategy_Returns'])
            
        # Check for Exit (Position goes to 0 while we were in a trade)
        elif pos == 0 and in_trade:
            in_trade = False
            exit_date = date
            exit_price = close_price
            
            cum_trade_ret = np.prod([1 + r for r in trade_returns]) - 1
            
            trades.append({
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Entry Price (Approx)': entry_price,
                'Exit Price': exit_price,
                'Duration (Days)': len(trade_returns),
                'Trade PnL': cum_trade_ret
            })
            trade_returns = []

    if in_trade:
        cum_trade_ret = np.prod([1 + r for r in trade_returns]) - 1
        trades.append({
            'Entry Date': entry_date,
            'Exit Date': df.index[-1],
            'Entry Price (Approx)': entry_price,
            'Exit Price': df.iloc[-1]['Close'],
            'Duration (Days)': len(trade_returns),
            'Trade PnL': cum_trade_ret
        })

    return pd.DataFrame(trades)

# --- Main Logic ---

st.title("🚀 Leveraged HMM-SVR Algo Trader")
st.markdown("""
**The "Kill Switch" Strategy:**
1. **Destined to Lose?** If HMM detects "Crash Mode", we **CASH OUT** instantly (0% exposure).
2. **Destined to Win?** If SVR predicts Low Volatility (Safe), we **LEVERAGE UP** based on confidence.
""")

# Sidebar Inputs
with st.sidebar:
    st.header("Settings")
    
    ticker = st.selectbox(
        "Ticker", 
        ["BTC-USD", "BNB-USD","ETH-USD","SOL-USD"],
        key="ticker_select" 
    )
    
    # --- NEW LEVERAGE INPUT ---
    st.markdown("### ⚡ Leverage Controls")
    max_leverage = st.slider("Max Leverage Multiplier", 1.0, 5.0, 2.0, step=0.5, help="If the model is super confident, how much leverage to use? (e.g., 3.0 = 3x leverage)")
    # ---------------------------

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
    
    st.caption("Note: Models will automatically train on the **4 years** of data prior to your selected Start Date.")
    
    st.divider()
    
    short_window = st.number_input("Fast EMA", 12, key="fast_ema")
    long_window = st.number_input("Slow EMA", 26, key="slow_ema")
    n_states = st.slider("HMM States", 2, 4, 3, key="hmm_slider")

if st.button("Run Leveraged Backtest"):
    train_start_date = pd.Timestamp(backtest_start) - pd.DateOffset(years=4)
    
    df = fetch_data(ticker, train_start_date, backtest_end)
    
    if df is None or len(df) < 200:
        st.error(f"Not enough data found for {ticker}. Ensure the ticker existed 4 years prior to {backtest_start}.")
    else:
        # 1. Feature Engineering
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = df['Log_Returns'].rolling(window=10).std()
        
        df['Downside_Returns'] = df['Log_Returns'].apply(lambda x: x if x < 0 else 0)
        df['Downside_Vol'] = df['Downside_Returns'].rolling(window=10).std()
        
        df['EMA_Short'] = df['Close'].ewm(span=short_window, adjust=False).mean()
        df['EMA_Long'] = df['Close'].ewm(span=long_window, adjust=False).mean()
        
        df['Target_Next_Vol'] = df['Volatility'].shift(-1)
        
        df = df.dropna()
        
        # 2. Split Data
        train_df = df[df.index < pd.Timestamp(backtest_start)].copy()
        test_df = df[df.index >= pd.Timestamp(backtest_start)].copy()
        
        if len(train_df) < 365:
            st.warning(f"Warning: Only {len(train_df)} days found for training. HMM performs best with >2 years of data.")
        
        if len(test_df) < 10:
             st.error("Not enough data for backtesting range.")
        else:
            st.info(f"Training on {len(train_df)} days. Backtesting on {len(test_df)} days.")
            
            with st.spinner("Training HMM (Crash Detection)..."):
                hmm_model, state_map = train_hmm_model(train_df, n_states)
                
                X_train_hmm = train_df[['Log_Returns', 'Volatility']].values * 100
                train_raw_states = hmm_model.predict(X_train_hmm)
                train_df['Regime'] = [state_map.get(s, s) for s in train_raw_states]
                
            with st.spinner("Training SVR (Volatility Prediction)..."):
                svr_model, svr_scaler = train_svr_model(train_df)
                
            with st.spinner("Running Leverage Logic..."):
                # --- OUT OF SAMPLE BACKTEST ---
                
                X_test_hmm = test_df[['Log_Returns', 'Volatility']].values * 100
                test_raw_states = hmm_model.predict(X_test_hmm)
                test_df['Regime'] = [state_map.get(s, s) for s in test_raw_states]
                
                X_test_svr = test_df[['Log_Returns', 'Volatility', 'Downside_Vol', 'Regime']].values
                X_test_svr_scaled = svr_scaler.transform(X_test_svr)
                test_df['Predicted_Vol'] = svr_model.predict(X_test_svr_scaled)
                
                high_vol_state = n_states - 1
                
                test_df['Signal'] = np.where(test_df['EMA_Short'] > test_df['EMA_Long'], 1, 0)
                
                # --- NEW LEVERAGE LOGIC ---
                avg_train_vol = train_df['Volatility'].mean()
                
                # Risk Ratio: < 1.0 means safer than average, > 1.0 means riskier
                test_df['Risk_Ratio'] = test_df['Predicted_Vol'] / avg_train_vol
                
                # Inverse Volatility Sizing: 
                # If Risk is 0.5 (very safe), Size becomes 2.0 (2x leverage)
                test_df['Position_Size'] = (1.0 / test_df['Risk_Ratio'])
                
                # Cap the leverage at user setting (e.g., 3x)
                test_df['Position_Size'] = test_df['Position_Size'].clip(upper=max_leverage, lower=0.0)
                
                # --- KILL SWITCH (Destined to Lose) ---
                test_df['Position_Size'] = np.where(
                    test_df['Regime'] == high_vol_state,
                    0.0, # CASH OUT
                    test_df['Position_Size']
                )
                
                test_df['Final_Position'] = (test_df['Signal'] * test_df['Position_Size']).shift(1)
                
                test_df['Simple_Returns'] = test_df['Close'].pct_change()
                test_df['Strategy_Returns'] = test_df['Final_Position'] * test_df['Simple_Returns']
                test_df['Buy_Hold_Returns'] = test_df['Simple_Returns']
                
                test_df['Strategy_Value'] = (1 + test_df['Strategy_Returns'].fillna(0)).cumprod()
                test_df['Buy_Hold_Value'] = (1 + test_df['Buy_Hold_Returns'].fillna(0)).cumprod()
                
                test_df.dropna(inplace=True)
                
                # --- EXTRACT TRADES ---
                trade_log = generate_trade_log(test_df)
                
                # --- RESULTS ---
                
                metrics_df = calculate_metrics(test_df)
                st.subheader(f"Performance Metrics (Max Leverage: {max_leverage}x)")
                st.table(metrics_df)
                
                # Charts
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Equity Curve")
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Buy_Hold_Value'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
                    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Strategy_Value'], name='Leveraged Strategy', line=dict(color='#00CC96', width=2)))
                    
                    # Add Markers
                    if not trade_log.empty:
                        buy_points = trade_log.set_index('Entry Date')
                        buy_vals = test_df.loc[buy_points.index]['Strategy_Value']
                        sell_points = trade_log.set_index('Exit Date')
                        sell_vals = test_df.loc[sell_points.index]['Strategy_Value']

                        fig.add_trace(go.Scatter(x=buy_points.index, y=buy_vals, mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=10, color='lime')))
                        fig.add_trace(go.Scatter(x=sell_points.index, y=sell_vals, mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=10, color='red')))

                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    st.subheader("Active Leverage Deployment")
                    st.caption("When line > 1.0, you are using Leverage.")
                    
                    # Color the area to show leverage intensity
                    fig_size = px.area(test_df, x=test_df.index, y='Position_Size', title="Leverage Over Time")
                    fig_size.add_hline(y=1.0, line_dash="dash", line_color="white", annotation_text="1x Baseline")
                    st.plotly_chart(fig_size, use_container_width=True)
                
                st.divider()
                st.subheader("📝 Trade Log")
                if not trade_log.empty:
                    display_log = trade_log.copy()
                    display_log['Entry Date'] = display_log['Entry Date'].dt.date
                    display_log['Exit Date'] = display_log['Exit Date'].dt.date
                    display_log['Trade PnL'] = display_log['Trade PnL'].map('{:.2%}'.format)
                    display_log['Entry Price (Approx)'] = display_log['Entry Price (Approx)'].map('{:.2f}'.format)
                    display_log['Exit Price'] = display_log['Exit Price'].map('{:.2f}'.format)
                    st.dataframe(display_log, use_container_width=True)
                else:
                    st.write("No trades executed.")

                st.subheader("SVR Prediction Accuracy")
                fig_svr = go.Figure()
                slice_df = test_df.iloc[-100:] 
                fig_svr.add_trace(go.Scatter(x=slice_df.index, y=slice_df['Target_Next_Vol'], name='Actual Volatility'))
                fig_svr.add_trace(go.Scatter(x=slice_df.index, y=slice_df['Predicted_Vol'], name='Predicted Volatility', line=dict(dash='dot')))
                st.plotly_chart(fig_svr, use_container_width=True)
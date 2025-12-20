import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date
import io

# --- Config ---
st.set_page_config(page_title="HMM-SVR Leverage Sniper", layout="wide")

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

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

def calculate_metrics(df, strategy_col='Strategy_Value', benchmark_col='Buy_Hold_Value', leverage_3x_col='Buy_Hold_3x_Value'):
    stats = {}
    cols_to_process = [
        (strategy_col, 'Regime-Conditional Volatility Suppression'), 
        (benchmark_col, 'Buy & Hold 1x'),
        (leverage_3x_col, 'Buy & Hold 3x')
    ]
    
    for col, name in cols_to_process:
        if col not in df.columns:
            continue
            
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

def create_research_grade_equity_plot(df, ticker):
    """Create publication-quality equity curve plot with log scale"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors
    colors = {
        'strategy': '#2ecc71',  # Green
        'hold_1x': '#95a5a6',   # Gray
        'hold_3x': '#e74c3c'    # Red
    }
    
    # Plot equity curves
    ax.plot(df.index, df['Buy_Hold_Value'], 
            label='Buy & Hold 1x', color=colors['hold_1x'], 
            linestyle='--', linewidth=2, alpha=0.7)
    
    ax.plot(df.index, df['Buy_Hold_3x_Value'], 
            label='Buy & Hold 3x', color=colors['hold_3x'], 
            linestyle='-.', linewidth=2.5, alpha=0.8)
    
    ax.plot(df.index, df['Strategy_Value'], 
            label='Regime-Conditional Volatility Suppression', color=colors['strategy'], 
            linewidth=3, alpha=0.9)
    
    # Calculate final returns for title
    final_strategy = (df['Strategy_Value'].iloc[-1] - 1) * 100
    final_1x = (df['Buy_Hold_Value'].iloc[-1] - 1) * 100
    final_3x = (df['Buy_Hold_3x_Value'].iloc[-1] - 1) * 100
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Formatting
    ax.set_title(f'{ticker.replace("-USD", "")} Strategy Performance (Log Scale)\n'
                 f'RCVS: {final_strategy:.1f}% | 1x: {final_1x:.1f}% | 3x: {final_3x:.1f}%', 
                 fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Portfolio Value (Log Scale, Normalized to 1.0)', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, which='both')
    ax.legend(loc='best', framealpha=0.95, edgecolor='black')
    ax.axhline(y=1, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add subtle background
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    return fig

def create_research_grade_drawdown_plot(df, ticker):
    """Create publication-quality drawdown plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate drawdowns
    drawdowns = {}
    for col, name in [('Strategy_Value', 'Regime-Conditional Volatility Suppression'),
                      ('Buy_Hold_Value', 'Buy & Hold 1x'),
                      ('Buy_Hold_3x_Value', 'Buy & Hold 3x')]:
        rolling_max = df[col].cummax()
        drawdown = (df[col] - rolling_max) / rolling_max * 100
        drawdowns[name] = drawdown
    
    # Define colors
    colors = {
        'Regime-Conditional Volatility Suppression': '#2ecc71',
        'Buy & Hold 1x': '#95a5a6',
        'Buy & Hold 3x': '#e74c3c'
    }
    
    # Plot drawdowns with filled areas
    for name, dd in drawdowns.items():
        ax.fill_between(df.index, 0, dd, 
                        label=name, color=colors[name], alpha=0.4)
        ax.plot(df.index, dd, color=colors[name], linewidth=1.5, alpha=0.8)
    
    # Calculate max drawdowns for title
    max_dds = {name: dd.min() for name, dd in drawdowns.items()}
    
    # Formatting
    ax.set_title(f'{ticker.replace("-USD", "")} Drawdown Analysis\n'
                 f'RCVS: {max_dds["Regime-Conditional Volatility Suppression"]:.1f}% | '
                 f'1x: {max_dds["Buy & Hold 1x"]:.1f}% | '
                 f'3x: {max_dds["Buy & Hold 3x"]:.1f}%', 
                 fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='lower left', framealpha=0.95, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add subtle background
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    return fig

def create_research_grade_metrics_table(df, ticker):
    """Create publication-quality metrics table"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate metrics
    metrics_data = []
    strategies = [
        ('Strategy_Value', 'Regime-Conditional Volatility Suppression'),
        ('Buy_Hold_Value', 'Buy & Hold 1x'),
        ('Buy_Hold_3x_Value', 'Buy & Hold 3x')
    ]
    
    for col, name in strategies:
        initial = df[col].iloc[0]
        final = df[col].iloc[-1]
        total_return = (final - initial) / initial * 100
        
        daily_ret = df[col].pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365) if daily_ret.std() != 0 else 0
        
        rolling_max = df[col].cummax()
        drawdown = (df[col] - rolling_max) / rolling_max * 100
        max_dd = drawdown.min()
        
        # Calculate win rate
        if 'Strategy_Returns' in df.columns and col == 'Strategy_Value':
            wins = (df['Strategy_Returns'] > 0).sum()
            total_trades = (df['Strategy_Returns'] != 0).sum()
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        else:
            returns = df[col].pct_change().dropna()
            wins = (returns > 0).sum()
            win_rate = (wins / len(returns) * 100) if len(returns) > 0 else 0
        
        metrics_data.append([
            name,
            f"{total_return:.2f}%",
            f"{sharpe:.2f}",
            f"{max_dd:.2f}%",
            f"{win_rate:.1f}%"
        ])
    
    # Create table
    columns = ['Strategy', 'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    table = ax.table(cellText=metrics_data, 
                     colLabels=columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
        cell.set_edgecolor('white')
    
    # Style rows with alternating colors
    row_colors = ['#e8f8f5', '#f8f9fa', '#fff5f5']
    for i in range(1, len(metrics_data) + 1):
        for j in range(len(columns)):
            cell = table[(i, j)]
            cell.set_facecolor(row_colors[i-1])
            cell.set_edgecolor('#cccccc')
            if j == 0:  # Strategy name column
                cell.set_text_props(weight='bold')
    
    plt.title(f'{ticker.replace("-USD", "")} Performance Metrics', 
              fontsize=14, fontweight='bold', pad=20)
    
    return fig

def fig_to_buffer(fig):
    """Convert matplotlib figure to buffer for download"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return buf

# --- Main Logic ---

st.title("⚡ HMM-SVR Leverage Backtester")
st.markdown("""
**The "Strict Rules" Strategy (No Lookahead Bias):**
1. **Baseline:** Buy when Fast EMA > Slow EMA.
2. **Safety (HMM):** Calculates market regime using ONLY past data.
3. **Leverage Boost:** Uses SVR to predict *tomorrow's* volatility based on *today's* data.
**Timing:** Uses End-of-Day (EOD) data to make decisions for the next trading day.
**Benchmark:** Compares against 1x Buy & Hold AND 3x Leveraged Buy & Hold.
""")

with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox("Ticker", ["BNB-USD", "ETH-USD", "SOL-USD", "LINK-USD", "BTC-USD"])
    backtest_start = st.date_input("Backtest Start Date", date(2022, 1, 1))
    backtest_end = st.date_input("Backtest End Date", datetime.now())
    st.divider()
    st.subheader("Leverage Rules")
    leverage_mult = st.number_input("Boost Leverage", value=3.0, step=0.5)
    risk_threshold = st.slider("Certainty Threshold", 0.1, 1.0, 0.5)
    st.divider()
    st.subheader("📊 Research-Grade Charts")
    generate_research_plots = st.checkbox("Generate Publication Charts", value=True, 
                                          help="Creates high-quality matplotlib charts suitable for research papers")

if st.button("Run Honest Backtest", type="primary"):
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
            
            lookback_window = 252 
            
            for i in range(total_steps):
                # Update UI
                if i % 10 == 0: progress_bar.progress((i + 1) / total_steps)
                
                curr_pointer = start_idx + i
                window_start = max(0, curr_pointer - lookback_window)
                history_slice = all_data.iloc[window_start : curr_pointer + 1]
                
                # --- A. Honest Regime Detection ---
                X_slice = history_slice[['Log_Returns', 'Volatility']].values * 100
                
                try:
                    hidden_states_slice = hmm_model.predict(X_slice)
                    current_state_raw = hidden_states_slice[-1]
                    current_state = state_map.get(current_state_raw, current_state_raw)
                except:
                    current_state = 1
                
                honest_regimes.append(current_state)
                
                # --- B. Honest Volatility Prediction ---
                row = test_df.iloc[i]
                svr_features = np.array([[
                    row['Log_Returns'], 
                    row['Volatility'], 
                    row['Downside_Vol'], 
                    current_state
                ]])
                
                svr_feat_scaled = svr_scaler.transform(svr_features)
                pred_vol = svr_model.predict(svr_feat_scaled)[0]
                honest_predicted_vols.append(pred_vol)
                
                # Calculate EMAs
                test_df.loc[test_df.index[i], 'EMA_Short'] = history_slice['Close'].ewm(span=12).mean().iloc[-1]
                test_df.loc[test_df.index[i], 'EMA_Long'] = history_slice['Close'].ewm(span=26).mean().iloc[-1]
            
            test_df['Regime'] = honest_regimes
            test_df['Predicted_Vol'] = honest_predicted_vols
            
            progress_bar.empty()
            
            # --- STRATEGY LOGIC ---
            
            test_df['Signal'] = np.where(test_df['EMA_Short'] > test_df['EMA_Long'], 1, 0)
            avg_train_vol = train_df['Volatility'].mean()
            test_df['Risk_Ratio'] = test_df['Predicted_Vol'] / avg_train_vol
            
            test_df['Position_Size'] = 1.0
            
            cond_safe = (test_df['Regime'] == 0)
            cond_low_risk = (test_df['Risk_Ratio'] < risk_threshold)
            cond_crash = (test_df['Regime'] == (n_states - 1))
            
            test_df['Position_Size'] = np.where(cond_safe & cond_low_risk, leverage_mult, test_df['Position_Size'])
            test_df['Position_Size'] = np.where(cond_crash, 0.0, test_df['Position_Size'])
            
            # Calculate Returns
            test_df['Final_Position'] = (test_df['Signal'] * test_df['Position_Size']).shift(1)
            test_df['Simple_Returns'] = test_df['Close'].pct_change()
            test_df['Strategy_Returns'] = test_df['Final_Position'] * test_df['Simple_Returns']
            test_df['Buy_Hold_3x_Returns'] = test_df['Simple_Returns'] * 3.0
            
            # Metrics & Plots
            test_df['Strategy_Value'] = (1 + test_df['Strategy_Returns'].fillna(0)).cumprod()
            test_df['Buy_Hold_Value'] = (1 + test_df['Simple_Returns'].fillna(0)).cumprod()
            test_df['Buy_Hold_3x_Value'] = (1 + test_df['Buy_Hold_3x_Returns'].fillna(0)).cumprod()
            test_df.dropna(inplace=True)
            
            # --- DISPLAY RESULTS ---
            
            metrics_df = calculate_metrics(test_df)
            st.subheader("📊 Performance vs Benchmark")
            st.table(metrics_df)
            
            # Interactive Plotly Charts
            st.subheader("📈 Interactive Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Buy_Hold_Value'], 
                                    name='Buy & Hold 1x', line=dict(color='gray', dash='dot')))
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Buy_Hold_3x_Value'], 
                                    name='Buy & Hold 3x', line=dict(color='orange', dash='dash', width=2)))
            fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Strategy_Value'], 
                                    name='Regime-Conditional Volatility Suppression', line=dict(color='#00CC96', width=2)))
            fig.update_layout(
                yaxis_title='Portfolio Value',
                yaxis_type='log',
                xaxis_title='Date',
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- RESEARCH-GRADE CHARTS ---
            if generate_research_plots:
                st.divider()
                st.header("📄 Publication-Quality Charts")
                st.markdown("*These charts are suitable for research papers and academic publications*")
                
                # Create research-grade plots
                with st.spinner("Generating high-resolution research charts..."):
                    
                    # 1. Equity Curve
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader("Figure 1: Equity Curve Comparison")
                    equity_fig = create_research_grade_equity_plot(test_df, ticker)
                    st.pyplot(equity_fig)
                    
                    # Download button for equity curve
                    equity_buf = fig_to_buffer(equity_fig)
                    with col2:
                        st.download_button(
                            label="📥 Download PNG",
                            data=equity_buf,
                            file_name=f"{ticker}_equity_curve.png",
                            mime="image/png"
                        )
                    
                    plt.close(equity_fig)
                    
                    st.markdown("---")
                    
                    # 2. Drawdown Plot
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader("Figure 2: Drawdown Analysis")
                    drawdown_fig = create_research_grade_drawdown_plot(test_df, ticker)
                    st.pyplot(drawdown_fig)
                    
                    # Download button for drawdown
                    drawdown_buf = fig_to_buffer(drawdown_fig)
                    with col2:
                        st.download_button(
                            label="📥 Download PNG",
                            data=drawdown_buf,
                            file_name=f"{ticker}_drawdown.png",
                            mime="image/png"
                        )
                    
                    plt.close(drawdown_fig)
                    
                    st.markdown("---")
                    
                    # 3. Metrics Table
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader("Table 1: Performance Metrics")
                    metrics_fig = create_research_grade_metrics_table(test_df, ticker)
                    st.pyplot(metrics_fig)
                    
                    # Download button for metrics table
                    metrics_buf = fig_to_buffer(metrics_fig)
                    with col2:
                        st.download_button(
                            label="📥 Download PNG",
                            data=metrics_buf,
                            file_name=f"{ticker}_metrics_table.png",
                            mime="image/png"
                        )
                    
                    plt.close(metrics_fig)
                    
                    st.success("✅ Research-grade charts generated successfully!")
                    st.info("💡 **Tip:** All charts are 300 DPI and publication-ready. Download them using the buttons above.")
            
            st.divider()
            
            # Leverage Deployment
            st.subheader("⚡ Leverage Deployment")
            fig_lev = go.Figure()
            fig_lev.add_trace(go.Scatter(x=test_df.index, y=test_df['Position_Size'], 
                                        mode='lines', fill='tozeroy', name='Leverage', 
                                        line=dict(color='#636EFA')))
            st.plotly_chart(fig_lev, use_container_width=True)
            
            # Trade Log
            trade_log = generate_trade_log(test_df)
            st.subheader("📝 Trade Log")
            if not trade_log.empty:
                display_log = trade_log.copy()
                display_log['Trade PnL'] = display_log['Trade PnL'].map('{:.2%}'.format)
                st.dataframe(display_log, use_container_width=True)
            else:
                st.write("No trades generated.")
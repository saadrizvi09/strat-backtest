import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="Crypto Quant Backtester", layout="wide")
st.title("⚡ Quant/HFT Crypto Backtesting Engine")
st.markdown("""
> **Reference:** *From Prediction to Profit: A Comprehensive Review of Cryptocurrency Trading Strategies*
>
> This tool implements the evaluation metrics (Sharpe, VaR) and strategy comparisons (Active vs. Buy & Hold) 
> discussed in Section V of the review.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Asset & Data Parameters")
ticker_map = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Litecoin (LTC)": "LTC-USD"
}
selected_asset = st.sidebar.selectbox("Select Asset", list(ticker_map.keys()))
ticker = ticker_map[selected_asset]

start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", datetime.now())
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000.0)
commission = st.sidebar.number_input("Commission per Trade (%)", value=0.1) / 100.0

st.sidebar.header("2. Strategy Logic")
strategy_type = st.sidebar.selectbox(
    "Select Strategy", 
    ["SMA Crossover (Trend)", "RSI Mean Reversion (Momentum)"]
)

# Dynamic Parameters based on Strategy
params = {}
if strategy_type == "SMA Crossover (Trend)":
    params['short_window'] = st.sidebar.slider("Short Window (Fast MA)", 5, 50, 20)
    params['long_window'] = st.sidebar.slider("Long Window (Slow MA)", 20, 200, 50)
elif strategy_type == "RSI Mean Reversion (Momentum)":
    params['rsi_period'] = st.sidebar.slider("RSI Period", 5, 30, 14)
    params['overbought'] = st.sidebar.slider("Overbought Threshold", 50, 90, 70)
    params['oversold'] = st.sidebar.slider("Oversold Threshold", 10, 50, 30)

# --- BACKTESTING ENGINE ---
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    # Flatten MultiIndex columns if present (common in new yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_metrics(daily_returns, risk_free_rate=0.02):
    """Calculates metrics defined in PDF Table 6"""
    total_return = (1 + daily_returns).prod() - 1
    annualized_return = (1 + daily_returns.mean())**252 - 1
    volatility = daily_returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
    
    # Max Drawdown
    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    # Value at Risk (VaR) (95% confidence)
    var_95 = np.percentile(daily_returns, 5)
    
    return {
        "Total Return": total_return,
        "Ann. Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "VaR (95%)": var_95
    }

def run_backtest(df, strategy_type, params, initial_capital, commission):
    data = df.copy()
    data['Returns'] = data['Close'].pct_change()
    
    # 1. Generate Signals
    if strategy_type == "SMA Crossover (Trend)":
        data['Fast_MA'] = data['Close'].rolling(window=params['short_window']).mean()
        data['Slow_MA'] = data['Close'].rolling(window=params['long_window']).mean()
        data['Signal'] = 0
        data.iloc[params['short_window']:, data.columns.get_loc('Signal')] = np.where(
            data['Fast_MA'][params['short_window']:] > data['Slow_MA'][params['short_window']:], 1, 0
        )
        data['Position'] = data['Signal'].diff() # 1 = Buy, -1 = Sell

    elif strategy_type == "RSI Mean Reversion (Momentum)":
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=params['rsi_period']).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        data['Signal'] = 0
        # Buy when RSI < Oversold, Sell when RSI > Overbought
        data.loc[data['RSI'] < params['oversold'], 'Signal'] = 1 
        data.loc[data['RSI'] > params['overbought'], 'Signal'] = 0 # Exit
        data['Position'] = data['Signal'].diff()

    # 2. Simulate Trades
    balance = initial_capital
    holdings = 0
    portfolio_values = []
    
    # Vectorized simulation usually faster, but iterative allows easier fee handling for display
    # We will use a simplified iterative approach for clarity
    
    in_position = False
    entry_price = 0
    
    trade_log = []
    
    for index, row in data.iterrows():
        price = row['Close']
        action = row['Position']
        
        # Buy Signal
        if action == 1 and not in_position:
            cost = balance * (1 - commission)
            holdings = cost / price
            balance = 0
            in_position = True
            entry_price = price
            trade_log.append({'Date': index, 'Type': 'Buy', 'Price': price, 'Value': cost})
            
        # Sell Signal
        elif (action == -1 or (strategy_type == "RSI Mean Reversion (Momentum)" and row['Signal'] == 0)) and in_position:
            revenue = holdings * price * (1 - commission)
            balance = revenue
            holdings = 0
            in_position = False
            trade_log.append({'Date': index, 'Type': 'Sell', 'Price': price, 'Value': revenue})
            
        # Mark-to-market
        current_val = balance + (holdings * price)
        portfolio_values.append(current_val)
        
    data['Portfolio Value'] = portfolio_values
    data['Strategy Returns'] = data['Portfolio Value'].pct_change()
    
    return data, pd.DataFrame(trade_log)

# --- EXECUTION ---
if st.sidebar.button("Run Backtest"):
    try:
        with st.spinner('Fetching data and calculating strategies...'):
            df = load_data(ticker, start_date, end_date)
            
            if df.empty:
                st.error("No data found for selected range.")
            else:
                # Run Strategy
                results, trade_log = run_backtest(df, strategy_type, params, initial_capital, commission)
                
                # Buy & Hold Logic
                bh_returns = results['Returns']
                bh_metrics = calculate_metrics(bh_returns)
                
                # Strategy Logic
                strat_returns = results['Strategy Returns']
                strat_metrics = calculate_metrics(strat_returns)

                # --- DASHBOARD LAYOUT ---
                
                # 1. Metrics Comparison Table
                st.subheader("Performance Metrics (Table 6 Reference)")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Return (Strategy)", f"{strat_metrics['Total Return']:.2%}", 
                              delta=f"{(strat_metrics['Total Return'] - bh_metrics['Total Return']):.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{strat_metrics['Sharpe Ratio']:.2f}",
                              delta=f"{strat_metrics['Sharpe Ratio'] - bh_metrics['Sharpe Ratio']:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{strat_metrics['Max Drawdown']:.2%}")
                with col4:
                    st.metric("VaR (95%)", f"{strat_metrics['VaR (95%)']:.2%}")

                # 2. Equity Curve Comparison
                st.subheader("Equity Curve: Strategy vs. Buy & Hold")
                fig = go.Figure()
                
                # Strategy Equity
                fig.add_trace(go.Scatter(x=results.index, y=results['Portfolio Value'], 
                                         mode='lines', name='Active Strategy', line=dict(color='green', width=2)))
                
                # Buy & Hold Equity (Normalized to Initial Capital)
                bh_value = (1 + results['Returns']).cumprod() * initial_capital
                fig.add_trace(go.Scatter(x=results.index, y=bh_value, 
                                         mode='lines', name='Buy & Hold', line=dict(color='gray', dash='dash')))
                
                # Buy/Sell Markers
                if not trade_log.empty:
                    buys = trade_log[trade_log['Type'] == 'Buy']
                    sells = trade_log[trade_log['Type'] == 'Sell']
                    
                    fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Value'], mode='markers', 
                                             name='Buy Signal', marker=dict(color='blue', symbol='triangle-up', size=10)))
                    fig.add_trace(go.Scatter(x=sells['Date'], y=sells['Value'], mode='markers', 
                                             name='Sell Signal', marker=dict(color='red', symbol='triangle-down', size=10)))

                fig.update_layout(template="plotly_dark", height=500, xaxis_title="Date", yaxis_title="Portfolio Value ($)")
                st.plotly_chart(fig, use_container_width=True)

                # 3. Detailed Price & Signals
                st.subheader("Price Action & Indicators")
                fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                
                # Candlestick
                fig2.add_trace(go.Candlestick(x=results.index, open=results['Open'], high=results['High'],
                                              low=results['Low'], close=results['Close'], name='Price'), row=1, col=1)
                
                # Indicator Overlay
                if strategy_type == "SMA Crossover (Trend)":
                    fig2.add_trace(go.Scatter(x=results.index, y=results['Fast_MA'], name='Fast MA', line=dict(color='orange')), row=1, col=1)
                    fig2.add_trace(go.Scatter(x=results.index, y=results['Slow_MA'], name='Slow MA', line=dict(color='purple')), row=1, col=1)
                elif strategy_type == "RSI Mean Reversion (Momentum)":
                    fig2.add_trace(go.Scatter(x=results.index, y=results['RSI'], name='RSI', line=dict(color='cyan')), row=2, col=1)
                    fig2.add_hline(y=params['overbought'], line_dash="dot", row=2, col=1, annotation_text="Overbought")
                    fig2.add_hline(y=params['oversold'], line_dash="dot", row=2, col=1, annotation_text="Oversold")

                fig2.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
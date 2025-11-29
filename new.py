import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import tensorflow as tf
from datetime import datetime

# Set page config
st.set_page_config(page_title="SAM-LSTM Smart Backtester", layout="wide")

# Constants matching training script
GROUPS = {
    "Price": 4, "Adoption": 10, "Distribution": 11, "Market": 14, "Valuation": 3
}
SEQ_LEN = 7

# --- Custom Layers for Model Loading ---
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)

# --- Helper Functions ---

@st.cache_resource
def load_trained_model(uploaded_file):
    with open("temp_model.keras", "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tf.keras.models.load_model("temp_model.keras", custom_objects={'AttentionLayer': AttentionLayer}, compile=False)

def fetch_yfinance_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(df) == 0: return None
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs('Close', level=0, axis=1, drop_level=False)
            df.columns = ['Close']
        else:
            df = df[['Close']]
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def prepare_smart_features(price_df, custom_csv):
    """
    Intelligently maps Free Data metrics to SAM-LSTM groups.
    Fills missing data with 0 (Neutral) to avoid confusing the model.
    """
    try:
        onchain_df = pd.read_csv(custom_csv, index_col=0, parse_dates=True)
        onchain_df.index = onchain_df.index.tz_localize(None)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    # Merge
    merged_df = price_df.join(onchain_df, how='inner').ffill()
    if len(merged_df) < SEQ_LEN:
        st.error("Not enough data.")
        return None

    # Normalize (Standard Score)
    norm_df = (merged_df - merged_df.mean()) / merged_df.std()
    
    # MAPPING STRATEGY
    data = {}
    
    # 1. PRICE GROUP (4 slots)
    p_feat = norm_df['Close'].values
    data["Price"] = np.column_stack([p_feat, p_feat, p_feat, p_feat])

    # 2. ADOPTION GROUP (10 slots)
    a1 = norm_df.get('Adoption_ActiveAddresses', pd.Series(0, index=norm_df.index)).values
    a2 = norm_df.get('Adoption_TxCount', pd.Series(0, index=norm_df.index)).values
    zeros_adoption = np.zeros((len(norm_df), 8))
    data["Adoption"] = np.column_stack([a1, a2, zeros_adoption])

    # 3. DISTRIBUTION GROUP (11 slots)
    d1 = norm_df.get('Distribution_MinerRevenue', pd.Series(0, index=norm_df.index)).values
    zeros_dist = np.zeros((len(norm_df), 10))
    data["Distribution"] = np.column_stack([d1, zeros_dist])

    # 4. MARKET GROUP (14 slots)
    m1 = norm_df.get('Market_HashRate', pd.Series(0, index=norm_df.index)).values
    m2 = norm_df.get('Market_Difficulty', pd.Series(0, index=norm_df.index)).values
    zeros_mkt = np.zeros((len(norm_df), 12))
    data["Market"] = np.column_stack([m1, m2, zeros_mkt])

    # 5. VALUATION GROUP (3 slots)
    v1 = norm_df.get('Valuation_MVRV', pd.Series(0, index=norm_df.index)).values
    zeros_val = np.zeros((len(norm_df), 2))
    data["Valuation"] = np.column_stack([v1, zeros_val])
    
    return data, merged_df

def generate_inference_features(price_df, noise_level=0.1):
    df = price_df.copy()
    data = {}
    price_norm = (df['Close'] - df['Close'].mean()) / df['Close'].std()
    for group, num_features in GROUPS.items():
        group_data = []
        for i in range(num_features):
            corr = 1 if i % 2 == 0 else -1 
            base_signal = price_norm * corr
            noise = np.random.normal(0, noise_level, len(df))
            feat = base_signal + noise
            group_data.append(feat)
        data[group] = np.array(group_data).T
    return data

def prepare_inference_sequences(feature_dict, start_idx=0):
    X_data = {g: [] for g in GROUPS}
    valid_indices = []
    first_group = list(feature_dict.keys())[0]
    n_rows = len(feature_dict[first_group])
    for i in range(start_idx, n_rows - SEQ_LEN):
        for group in GROUPS:
            window = feature_dict[group][i : i+SEQ_LEN]
            X_data[group].append(window)
        valid_indices.append(i + SEQ_LEN)
    return [np.array(X_data[g]) for g in GROUPS], valid_indices

def run_backtest(df, predictions, initial_capital, fee_pct, buy_threshold, sell_threshold, use_trend_filter=False):
    cash = initial_capital
    position = 0 
    equity_curve = []
    trade_log = []
    
    df = df.iloc[:len(predictions)].copy()
    
    # Calculate Trend Filter (50-Day SMA)
    if use_trend_filter:
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    for i in range(len(predictions) - 1):
        if use_trend_filter and i < 50:
            equity_curve.append(initial_capital)
            continue
            
        current_price = df['Close'].iloc[i]
        predicted_price = predictions[i][0]
        date = df.index[i]
        
        # Trend Logic
        is_uptrend = False
        if use_trend_filter:
            sma = df['SMA_50'].iloc[i]
            if current_price > sma:
                is_uptrend = True
        
        # AI Logic
        buy_signal = predicted_price > (current_price * buy_threshold)
        sell_signal = predicted_price < (current_price * sell_threshold)
        
        # Hybrid Decision Logic
        final_buy = False
        final_sell = False
        
        if use_trend_filter:
            # BUY: AI says Buy OR (Price > SMA and AI is not forcing sell)
            # We stick to AI for entry to catch dips, but use Trend to HOLD.
            final_buy = buy_signal
            
            # SELL: AI says Sell AND (Trend is Broken OR AI is super bearish?)
            # Protection: If in Strong Uptrend (Price > SMA), IGNORE AI Sell (HODL)
            if sell_signal:
                if is_uptrend:
                    final_sell = False # Ignore sell signal in uptrend (Let winners run)
                else:
                    final_sell = True
        else:
            final_buy = buy_signal
            final_sell = sell_signal
        
        # Execution
        if final_buy and position == 0:
            position = (cash * (1 - fee_pct)) / current_price
            cash = 0
            trade_log.append({'Date': date, 'Type': 'BUY', 'Price': current_price})
            
        elif final_sell and position > 0:
            cash = position * current_price * (1 - fee_pct)
            position = 0
            trade_log.append({'Date': date, 'Type': 'SELL', 'Price': current_price})
            
        current_equity = cash + (position * current_price)
        equity_curve.append(current_equity)
        
    return equity_curve, trade_log

# --- Main App Layout ---

st.title("🧠 SAM-LSTM Smart Backtester")

st.sidebar.header("1. Model & Data")
model_file = st.sidebar.file_uploader("Upload 'sam_lstm_model.keras'", type=["keras", "h5"])
ticker = st.sidebar.text_input("Ticker", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", datetime(2017, 1, 1))

data_source = st.sidebar.radio("Data Source", ["Real (CSV)", "Simulation"])

custom_csv = None
noise_level = 0.5

if data_source == "Real (CSV)":
    st.sidebar.info("Upload 'real_data_free.csv'")
    custom_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"])
else:
    noise_level = st.sidebar.slider("Simulation Noise", 0.0, 2.0, 0.5)

st.sidebar.divider()
st.sidebar.header("2. Strategy Settings")

# Hybrid Mode Switch
use_trend_filter = st.sidebar.checkbox("✅ Hybrid Mode (Trend Filter)", value=True, 
                                       help="If ON: Ignores Sell signals when Bitcoin is in a strong uptrend (Price > 50 SMA). Fixes 'early selling' in bull markets.")

# Debug Mode Switch
show_debug = st.sidebar.checkbox("🕵️‍♀️ Show Decision Debugger", value=False,
                                 help="Plot the Model's raw prediction confidence to help tune thresholds.")

buy_threshold = st.sidebar.slider("Buy Threshold", 0.95, 1.05, 1.00, step=0.001)
sell_threshold = st.sidebar.slider("Sell Threshold", 0.95, 1.05, 1.00, step=0.001)

initial_capital = st.sidebar.number_input("Capital", 10000)
fee_pct = 0.001

if model_file and ticker:
    try:
        model = load_trained_model(model_file)
        st.sidebar.success("Model Loaded!")
    except:
        st.stop()

    with st.spinner("Fetching Prices..."):
        price_df = fetch_yfinance_data(ticker, start_date, datetime.today())
    
    if price_df is not None:
        features_dict = None
        working_df = None
        
        if data_source == "Real (CSV)" and custom_csv:
            result = prepare_smart_features(price_df, custom_csv)
            if result:
                features_dict, working_df = result
        elif data_source == "Simulation":
            features_dict = generate_inference_features(price_df, noise_level)
            working_df = price_df

        if features_dict and working_df is not None:
            if st.button("Run Smart Backtest"):
                X_inputs, valid_indices = prepare_inference_sequences(features_dict)
                aligned_df = working_df.iloc[valid_indices].copy()
                
                preds = model.predict(X_inputs)
                
                equity, logs = run_backtest(aligned_df, preds, initial_capital, fee_pct, buy_threshold, sell_threshold, use_trend_filter)
                
                start_p = aligned_df['Close'].iloc[0]
                bh = (aligned_df['Close'] / start_p) * initial_capital
                min_len = min(len(equity), len(bh))
                
                res_df = pd.DataFrame({
                    'Strategy': equity[:min_len],
                    'Buy & Hold': bh.iloc[:min_len].values
                }, index=aligned_df.index[:min_len])
                
                st.line_chart(res_df)
                
                final = equity[-1]
                ret = (final - initial_capital) / initial_capital * 100
                st.metric("Final Equity", f"${final:,.2f}", f"{ret:.2f}%")
                st.metric("Total Trades", len(logs))
                
                # --- Decision Debugger Plot ---
                if show_debug:
                    aligned_df['Predicted'] = preds
                    # Calculate predicted % return
                    aligned_df['Pred_Pct_Change'] = (aligned_df['Predicted'] - aligned_df['Close']) / aligned_df['Close']
                    
                    st.divider()
                    st.subheader("🕵️‍♀️ Decision Debugger")
                    st.markdown("Use this to tune your **Buy Threshold**. If the blue line is below the red line, the model WON'T buy.")
                    
                    debug_fig = go.Figure()
                    # Plot 1: Model Confidence (Predicted Return)
                    debug_fig.add_trace(go.Scatter(
                        x=aligned_df.index, 
                        y=aligned_df['Pred_Pct_Change'], 
                        name="Model Confidence (Predicted %)",
                        line=dict(color='cyan', width=1)
                    ))
                    
                    # Plot 2: Buy Threshold Line
                    # The slider is e.g., 1.01 (101%). We subtract 1 to get percentage (0.01 or 1%)
                    thresh_val = buy_threshold - 1.0
                    debug_fig.add_trace(go.Scatter(
                        x=aligned_df.index, 
                        y=[thresh_val] * len(aligned_df), 
                        name="Buy Threshold",
                        line=dict(color='red', dash='dash')
                    ))
                    
                    debug_fig.update_layout(
                        title="Why is it buying (or not)?",
                        yaxis_title="Predicted Return (%)",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(debug_fig, use_container_width=True)
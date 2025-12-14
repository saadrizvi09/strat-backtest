import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Page Config ---
st.set_page_config(page_title="ML Trading Bot", layout="wide")
st.title("🤖 ML Alpha Strategy vs. Buy & Hold")
st.markdown("""
This app implements a **Hudson River Trading style** strategy.
It learns from historical data (**Training Period**) and then trades live from your Start Date (**Test Period**).
""")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="BTC-USD")

# The user selects when the "Backtest" (Trading) should actually start
test_start_date = st.sidebar.date_input("Strategy Start Date (Testing)", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# How many years of history BEFORE the start date should the model read to learn?
train_years = st.sidebar.slider("Training History (Years)", min_value=1, max_value=5, value=2)

interval = st.sidebar.selectbox("Interval", ["1h", "1d"], index=1)

# --- Helper Functions ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# --- Main Execution ---
if st.sidebar.button("Run Strategy"):
    with st.spinner('Fetching data and training model...'):
        
        # 1. Calculate the Real Fetch Start Date (Start Date - Training Years)
        real_start_date = pd.to_datetime(test_start_date) - pd.DateOffset(years=train_years)
        st.write(f"📥 Fetching data from **{real_start_date.date()}** to ensure model has {train_years} years of training context.")

        # 2. Fetch Data
        df = yf.download(ticker, start=real_start_date, end=end_date, interval=interval)
        
        # --- ROBUST FIX FOR YFINANCE & KEYERROR ---
        # 1. Drop Empty Columns if any
        df = df.dropna(axis=1, how='all')
        
        # 2. Flatten MultiIndex columns if they exist (e.g., ('Close', 'BTC-USD') -> 'Close')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 3. Ensure index is Datetime
        df.index = pd.to_datetime(df.index)
        # ------------------------------------------

        if len(df) < 100:
            st.error("Not enough data. Try a popular ticker like BTC-USD or SPY.")
        else:
            # 3. Feature Engineering
            df['Returns'] = df['Close'].pct_change()
            df['RSI'] = calculate_rsi(df)
            df['MACD'], df['Signal_Line'] = calculate_macd(df)
            df['Lagged_Return'] = df['Returns'].shift(1)
            
            # Create Target: 1 if price goes UP next period, 0 if DOWN
            df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
            
            # Drop NaNs created by indicators
            df = df.dropna()

            # Define Features (X) and Target (y)
            features = ['RSI', 'MACD', 'Signal_Line', 'Lagged_Return']
            X = df[features]
            y = df['Target']

            # 4. Strict Split based on User's Start Date
            # Convert user input to timestamp for comparison
            split_date = pd.Timestamp(test_start_date)
            
            # Train: Everything BEFORE the selected start date
            X_train = X[X.index < split_date]
            y_train = y[y.index < split_date]
            
            # Test: Everything AFTER the selected start date
            X_test = X[X.index >= split_date]
            y_test = y[y.index >= split_date]

            if len(X_test) == 0:
                st.error(f"No data found after {test_start_date}. Check your dates.")
            elif len(X_train) < 50:
                 st.error(f"Not enough training data before {test_start_date}. Increase 'Training History'.")
            else:
                # 5. Train Model
                model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
                model.fit(X_train, y_train)
                
                # 6. Make Predictions on Test Set
                preds = model.predict(X_test)
                
                # 7. Backtesting on Test Set
                test_data = df.loc[X_test.index].copy()
                test_data['Predicted_Signal'] = preds
                
                # Calculate Strategy Returns
                test_data['Strategy_Returns'] = test_data['Returns'] * test_data['Predicted_Signal']
                
                # Cumulative Returns
                test_data['Cumulative_Market'] = (1 + test_data['Returns']).cumprod()
                test_data['Cumulative_Strategy'] = (1 + test_data['Strategy_Returns']).cumprod()

                # --- Results Calculation ---
                market_return = test_data['Cumulative_Market'].iloc[-1] - 1
                strategy_return = test_data['Cumulative_Strategy'].iloc[-1] - 1
                accuracy = accuracy_score(y_test, preds)

                # --- Display ---
                st.subheader(f"Results (From {test_start_date} to {end_date})")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Buy & Hold Return", f"{market_return:.2%}")
                col2.metric("ML Strategy Return", f"{strategy_return:.2%}")
                col3.metric("Model Accuracy", f"{accuracy:.2%}")

                # Plotting
                st.subheader("Performance Comparison")
                chart_df = test_data[['Cumulative_Market', 'Cumulative_Strategy']]
                st.line_chart(chart_df)

                # Feature Importance
                st.subheader("What did the AI learn from the past?")
                feature_imp = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                st.bar_chart(feature_imp.set_index('Feature'))

else:
    st.info("Adjust settings in the sidebar and click 'Run Strategy' to start.")
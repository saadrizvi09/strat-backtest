import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import ruptures as rpt
from sklearn.preprocessing import StandardScaler
import os

# --- 1. Configuration & Mock Data Generation ---

# The paper groups 42 variables into 5 categories.
# We define the number of features per group based on the paper's description.
GROUPS = {
    "Price": 4,         # BTC, ETH, XRP, LTC prices
    "Adoption": 10,     # Active addresses, tx count, etc.
    "Distribution": 11, # Whale balances, exchange balances, etc.
    "Market": 14,       # Market cap, MVRV, etc.
    "Valuation": 3      # NVT ratio, etc.
}
SEQ_LEN = 7 # sliding window size (7 days)

def generate_mock_onchain_data(days=2000):
    """
    Generates synthetic data mimicking the structure of Glassnode/CryptoQuant data.
    In a real scenario, you would replace this with API calls.
    """
    print(f"Generating {days} days of synthetic on-chain data...")
    dates = pd.date_range(start="2018-01-01", periods=days, freq="D")
    
    # Generate a base price trend (Random Walk)
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, days)
    price = 10000 * np.cumprod(1 + returns)
    
    data = {"Date": dates, "Target_Price": price}
    
    # Generate features for each group correlated with price
    for group, num_features in GROUPS.items():
        for i in range(num_features):
            # Create features that are loosely correlated with price (some pos, some neg)
            noise = np.random.normal(0, 0.1, days)
            correlation = np.random.choice([1, -1])
            feature_data = (price * correlation) + (price * noise)
            data[f"{group}_{i}"] = feature_data
            
    return pd.DataFrame(data).set_index("Date")

# --- 2. Preprocessing (CPD & Sliding Window) ---

def apply_cpd_normalization(df, penalty=10):
    """
    Applies Change Point Detection (PELT) to segment the data.
    Normalizes each segment independently (Z-score).
    """
    print("Running PELT Change Point Detection...")
    price_signal = df["Target_Price"].values.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(price_signal)
    change_points = algo.predict(pen=penalty)
    
    df_norm = df.copy()
    start_idx = 0
    
    for end_idx in change_points:
        # Ensure we don't go out of bounds
        end_idx = min(end_idx, len(df))
        
        # Select segment
        segment = df.iloc[start_idx:end_idx]
        
        # Normalize this segment ONLY
        scaler = StandardScaler()
        # We normalize all columns except Date (which is index)
        df_norm.iloc[start_idx:end_idx] = scaler.fit_transform(segment)
        
        start_idx = end_idx
        
    return df_norm

def create_sequences(df_norm, df_raw):
    """
    Creates 5 separate input arrays (one for each group) and 1 target array.
    """
    X_data = {g: [] for g in GROUPS}
    y_data = []
    
    # Convert df to dictionary of group arrays
    group_arrays = {}
    for group in GROUPS:
        cols = [c for c in df_norm.columns if c.startswith(group)]
        group_arrays[group] = df_norm[cols].values
    
    target = df_raw["Target_Price"].values # We predict raw price (or return)
    
    for i in range(len(df_norm) - SEQ_LEN):
        # Input: Day i to i+SEQ_LEN
        for group in GROUPS:
            X_data[group].append(group_arrays[group][i : i+SEQ_LEN])
            
        # Target: Price at i+SEQ_LEN (Next Day)
        y_data.append(target[i + SEQ_LEN])
        
    return [np.array(X_data[g]) for g in GROUPS], np.array(y_data)

# --- 3. Model Architecture (SAM-LSTM) ---

class AttentionLayer(layers.Layer):
    """
    Custom Bahdanau-style Self-Attention Layer.
    Computes a context vector from the LSTM output sequence.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # x shape: (batch_size, seq_len, lstm_units)
        # Score computation e = tanh(W*x + b)
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        # Remove dimension of size 1
        e = tf.keras.backend.squeeze(e, axis=-1)
        # Compute weights
        alpha = tf.keras.backend.softmax(e)
        # Reshape to (batch, seq_len, 1)
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)
        # Context vector = sum(alpha * x)
        context = x * alpha
        context = tf.keras.backend.sum(context, axis=1)
        return context

def build_sam_lstm_model():
    """
    Constructs the Multi-Input LSTM with Attention.
    Structure: 5 Inputs -> 5 LSTMs -> 5 Attention Layers -> Concatenate -> Dense -> Output
    """
    inputs = []
    attention_outputs = []
    
    for group, num_features in GROUPS.items():
        # 1. Input Layer per group
        inp = Input(shape=(SEQ_LEN, num_features), name=f"Input_{group}")
        inputs.append(inp)
        
        # 2. LSTM Layer
        # return_sequences=True is required for Attention to look at all time steps
        lstm_out = layers.LSTM(32, return_sequences=True, name=f"LSTM_{group}")(inp)
        
        # 3. Attention Mechanism
        context_vector = AttentionLayer(name=f"Att_{group}")(lstm_out)
        attention_outputs.append(context_vector)
        
    # 4. Aggregation (MLP)
    concat = layers.Concatenate()(attention_outputs)
    dense1 = layers.Dense(64, activation="relu")(concat)
    dense2 = layers.Dense(32, activation="relu")(dense1)
    output = layers.Dense(1, activation="linear", name="Price_Prediction")(dense2)
    
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# --- 4. Main Execution ---

if __name__ == "__main__":
    # A. Get Data
    df_raw = generate_mock_onchain_data()
    
    # B. Preprocess
    df_norm = apply_cpd_normalization(df_raw, penalty=15)
    
    # C. Create Sequences
    X, y = create_sequences(df_norm, df_raw)
    
    # Split Train/Test
    split = int(len(y) * 0.8)
    X_train = [x[:split] for x in X]
    X_test = [x[split:] for x in X]
    y_train = y[:split]
    y_test = y[split:]
    
    # D. Build & Train Model
    print("Building SAM-LSTM Model...")
    model = build_sam_lstm_model()
    model.summary()
    
    print("Training Model (this may take a minute)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,  # Increase this for real training
        batch_size=32,
        verbose=1
    )
    
    # E. Save Model
    save_path = "sam_lstm_model.keras"
    model.save(save_path)
    print(f"\nSUCCESS! Model saved to {os.path.abspath(save_path)}")
    print("You can now run 'streamlit run backtester_app.py' and upload this file.")
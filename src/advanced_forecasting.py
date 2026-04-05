import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA

# ==============================
# 📁 PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==============================
# 🔧 STANDARDIZE COLUMNS
# ==============================
def standardize_columns(df):
    df.columns = df.columns.str.strip().str.lower()

    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])

    elif 'order date' in df.columns:
        df.rename(columns={'order date': 'order_date'}, inplace=True)
        df['order_date'] = pd.to_datetime(df['order_date'])

    elif 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'order_date'}, inplace=True)
        df['order_date'] = pd.to_datetime(df['order_date'])

    else:
        raise ValueError(f"order_date column not found: {df.columns}")

    return df


# ==============================
# 📊 LOAD HISTORICAL DATA
# ==============================
def load_data():
    file_path = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)
    df = standardize_columns(df)

    return df


# ==============================
# 📡 LOAD LIVE DATA
# ==============================
def load_live_dataset():
    live_path = os.path.join(BASE_DIR, "data", "processed", "live_data.csv")

    if not os.path.exists(live_path):
        print("⚠️ live_data.csv not found — skipping live data")
        return pd.DataFrame()

    live_df = pd.read_csv(live_path)

    if live_df.empty:
        print("⚠️ live_data.csv is empty")
        return pd.DataFrame()

    live_df = standardize_columns(live_df)

    return live_df


# ==============================
# 🔄 COMBINE DATA
# ==============================
def load_full_data():
    hist_df = load_data()
    live_df = load_live_dataset()

    if not live_df.empty:
        df = pd.concat([hist_df, live_df], ignore_index=True)
    else:
        df = hist_df.copy()

    df = df.sort_values("order_date")

    # 🚀 PRODUCTION OPTIMIZATION (VERY IMPORTANT)
    df = df.tail(100)

    return df


# ==============================
# 📈 TRAIN MODEL (STARTUP ONLY)
# ==============================
def train_model():
    print("📊 Loading and preparing data...")

    df = load_full_data()

    if 'sales' not in df.columns:
        raise ValueError(f"'sales' column missing. Available columns: {df.columns}")

    # Aggregate time series
    ts = df.groupby('order_date')['sales'].sum()

    # Ensure proper datetime index
    ts.index = pd.to_datetime(ts.index)

    print("🤖 Training ARIMA model...")

    try:
        model = ARIMA(ts, order=(1, 1, 0))  # lightweight + stable
        model_fit = model.fit()

        print("✅ Model trained successfully!")

        return model_fit

    except Exception as e:
        print("❌ Model training failed:", str(e))
        raise e


# ==============================
# 🔮 FORECAST
# ==============================
def get_forecast(model_fit, steps=30):
    if model_fit is None:
        raise ValueError("Model is not initialized")

    try:
        forecast = model_fit.forecast(steps=steps)
        return forecast.tolist()

    except Exception as e:
        print("❌ Forecast error:", str(e))
        raise e
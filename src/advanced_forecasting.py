import pandas as pd
import os
import numpy as np
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

    # 🚀 LIMIT DATA FOR PERFORMANCE
    df = df.tail(100)

    return df


# ==============================
# 📈 TRAIN MODEL (STARTUP ONLY)
# ==============================


def train_model(product="Product A"):
    df = load_full_data()

    # 🔥 FILTER PRODUCT
    if "product" in df.columns:
        df = df[df["product"] == product]

    if df.empty:
        raise ValueError(f"No data for {product}")

    ts = df.groupby('order_date')['sales'].sum()
    ts.index = pd.to_datetime(ts.index)

    ts = ts.asfreq('D').fillna(method='ffill')
    ts = ts.astype(float)

    # 🔍 DATA DIAGNOSTICS
    data_points = len(ts)
    unique_vals = ts.nunique()

    print(f"{product}: points={data_points}, unique={unique_vals}")

    # ==============================
    # 🧠 DECISION ENGINE
    # ==============================

    # 🚨 CASE 1: VERY POOR DATA
    if data_points < 10 or unique_vals <= 1:
        print(f"⚠️ Using MEAN fallback for {product}")

        value = ts.mean()

        class MeanModel:
            def forecast(self, steps=30):
                return np.array([value] * steps)

        return MeanModel()

    # ⚠️ CASE 2: LOW VARIATION
    elif unique_vals < 5:
        print(f"⚠️ Using SMOOTHING model for {product}")

        ts_smooth = ts.rolling(window=3, min_periods=1).mean()

        model = ARIMA(ts_smooth, order=(1,1,1))
        return model.fit()

    # ✅ CASE 3: GOOD DATA
    else:
        print(f"✅ Using ARIMA for {product}")

        model = ARIMA(ts, order=(2,1,2))
        return model.fit()

# ==============================
# 🔮 FORECAST
# ==============================
def get_forecast(model_fit, steps=30):
    forecast = model_fit.forecast(steps=steps)

    # 🔥 FIX: safe clipping (works for numpy + all cases)
    forecast = np.maximum(forecast, 0)

    return forecast.tolist()
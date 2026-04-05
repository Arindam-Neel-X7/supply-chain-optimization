from fastapi import FastAPI
from src.advanced_forecasting import train_model, get_forecast

app = FastAPI(title="Supply Chain Forecasting API 🚀")

# ✅ GLOBAL MODEL CACHE
model_cache = None


# ✅ TRAIN MODEL ON STARTUP (IMPORTANT)
@app.on_event("startup")
def load_model():
    global model_cache

    print("🚀 Training model on startup...")

    try:
        model_cache = train_model()
        print("✅ Model loaded successfully!")

    except Exception as e:
        print("❌ Error during model training:", str(e))


# ✅ HOME
@app.get("/")
def home():
    return {"message": "API is running 🚀"}


# ✅ FORECAST ENDPOINT (FAST ⚡)
@app.get("/forecast")
def forecast():
    global model_cache

    try:
        if model_cache is None:
            return {"status": "error", "message": "Model not loaded"}

        forecast = get_forecast(model_cache)

        return {
            "status": "success",
            "forecast": forecast
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"status": "error", "message": str(e)}


# ✅ HEALTH CHECK
@app.get("/health")
def health():
    return {"status": "healthy"}
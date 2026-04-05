from fastapi import FastAPI
from app.model import run_forecast

app = FastAPI(title="Supply Chain Forecasting API 🚀")


@app.get("/")
def home():
    return {
        "message": "API is running",
        "endpoints": {
            "/forecast": "Get predictions",
            "/health": "Check API status"
        }
    }


@app.get("/forecast")
def forecast():
    try:
        result = run_forecast()
        return {
            "status": "success",
            "forecast": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/health")
def health():
    return {"status": "ok"}
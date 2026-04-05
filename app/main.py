from fastapi import FastAPI
from src.advanced_forecasting import train_model, get_forecast

app = FastAPI(title="Supply Chain Forecasting API 🚀")

# ✅ GLOBAL MODEL CACHE
model_cache = {}

@app.on_event("startup")
def load_models():
    global model_cache

    products = ["Product A", "Product B", "Product C"]

    for product in products:
        try:
            print(f"🚀 Training model for {product}...")
            model_cache[product] = train_model(product)
        except Exception as e:
            print(f"❌ Failed for {product}: {e}")

    print("✅ Model loading complete")


# ✅ HOME
@app.get("/")
def home():
    return {"message": "API is running 🚀"}


# ✅ FORECAST ENDPOINT (FAST ⚡)
@app.get("/forecast")
def forecast(product: str = "Product A"):
    global model_cache

    try:
        if product not in model_cache:
            return {
                "status": "error",
                "message": f"Model not found for {product}"
            }

        model = model_cache[product]

        forecast = get_forecast(model)

        return {
            "status": "success",
            "product": product,
            "forecast": forecast
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ✅ HEALTH CHECK
@app.get("/health")
def health():
    return {"status": "healthy"}
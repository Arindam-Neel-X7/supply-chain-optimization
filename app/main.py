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
        print(f"🚀 Training model for {product}...")

        # 👉 🔥 PLACE IT HERE
        model_cache[product] = train_model(product)

    print("✅ All models loaded!")


# ✅ HOME
@app.get("/")
def home():
    return {"message": "API is running 🚀"}


# ✅ FORECAST ENDPOINT (FAST ⚡)
@app.get("/forecast")
def forecast(product: str = "Product A"):
    global model_cache

    if product not in model_cache:
        return {"status": "error", "message": "Invalid product"}

    model = model_cache[product]

    forecast = get_forecast(model)

    return {
        "status": "success",
        "product": product,
        "forecast": forecast
    }


# ✅ HEALTH CHECK
@app.get("/health")
def health():
    return {"status": "healthy"}
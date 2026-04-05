import os
import requests
import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime
from dotenv import load_dotenv
from requests.exceptions import RequestException


load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
CALENDARIFIC_API_KEY = os.getenv("CALENDARIFIC_API_KEY")

# WEATHER API

def get_weather_data(city="London", api_key=OPENWEATHER_API_KEY):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        data = requests.get(url).json()
        return {
            "temperature": data['main']['temp'],
            "humidity": data['main']['humidity']
        }

    except Exception as e:
        print("Weather API Error:", e)
        return {"temperature": 0, "humidity": 0}

# GOOGLE TRENDS

def get_google_trends(keyword="laptop"):
    try:
        pytrends = TrendReq()
        pytrends.build_payload([keyword])
        data = pytrends.interest_over_time()

        if not data.empty and keyword in data.columns:
            return data[keyword].iloc[-1]
        else:
            return 0
    except RequestException as e:
        print(f"⚠️ Google Trends API Error: {e}")
        return 0

# HOLIDAY API

def get_holiday_flag(api_key=CALENDARIFIC_API_KEY, country="IN"):
    try:
        url = "https://calendarific.com/api/v2/holidays"
        params = {
            "api_key": api_key,
            "country": country,
            "year": datetime.now().year
        }
        response = requests.get(url, params=params)
        data = response.json()
        holidays = data.get('response', {}).get('holidays', [])
        today = str(datetime.now().date())
        return 1 if any(
            h.get('date', {}).get('iso', '').startswith(today)
            for h in holidays
        ) else 0

    except Exception as e:
        print("Holiday API Error:", e)
        return 0

# WORLD BANK (ECONOMIC)

def get_economic_indicator():
    try:
        url = "https://api.worldbank.org/v2/country/IN/indicator/NY.GDP.MKTP.CD?format=json"
        response = requests.get(url)
        data = response.json()
        if data and len(data) > 1 and data[1]:
            return data[1][0].get('value', 0) or 0
        else:
            return 0
    except Exception as e:
        print("World Bank API Error:", e)
        return 0

# AMAZON DATA (RAPIDAPI)

def get_amazon_features(api_key=RAPIDAPI_KEY):
    try:
        url = "https://real-time-amazon-data.p.rapidapi.com/search"
        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com"
        }
        params = {"query": "laptop", "page": "1", "country": "US"}
        data = requests.get(url, headers=headers, params=params).json()
        products = data.get('data', {}).get('products', [])
        if not products:
            return {"price": 0, "rating": 0, "review_count": 0}
        product = products[0]
        return {
            "price": product.get('price', {}).get('value', 0),
            "rating": product.get('rating', 0),
            "review_count": product.get('reviews_count', 0)
        }

    except Exception as e:
        print("Amazon API Error:", e)
        return {"price": 0, "rating": 0, "review_count": 0}

# MASTER FUNCTION (DYNAMIC)

def get_real_time_features():
    weather = get_weather_data()
    trend = get_google_trends()
    holiday = get_holiday_flag()
    economy = get_economic_indicator()
    amazon = get_amazon_features()

    features = {
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "trend_score": trend,
        "is_holiday": holiday,
        "gdp": economy,
        "price": amazon["price"],
        "rating": amazon["rating"],
        "review_count": amazon["review_count"],
        "timestamp": datetime.now()
    }

    return pd.DataFrame([features])
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# LOAD DATA

def load_data():
    file_path = "../data/processed/cleaned_data.csv"
    df = pd.read_csv(file_path)
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

# PREPARE TIME SERIES

def prepare_data(df):
    daily_sales = df.groupby('order_date')['sales'].sum()
    daily_sales.index = pd.to_datetime(daily_sales.index)
    daily_sales = daily_sales.asfreq('D')  # D = Daily
    daily_sales = daily_sales.ffill()
    
    return daily_sales

# TRAIN ARIMA MODEL

def train_arima(data):
    print("\nTraining ARIMA Model...")
    model = ARIMA(data, order=(5, 1, 0))  # basic config
    model_fit = model.fit()
    print("Model trained successfully!")
    return model_fit

# FORECAST FUTURE VALUES

def forecast(model_fit, steps=30):
    print("\nForecasting future demand...")
    forecast_values = model_fit.forecast(steps=steps)
    return forecast_values

# PLOT RESULTS

def plot_forecast(data, forecast_values):
    plt.figure(figsize=(12, 6))
    data[-100:].plot(label="Actual Sales")
    # Create future dates
    future_dates = pd.date_range(
        start=data.index[-1],
        periods=len(forecast_values)+1,
        freq='D'
    )[1:]

    forecast_values.index = future_dates
    forecast_values.plot(label="Forecast", color='red')

    plt.title("Demand Forecast (ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid()
    plt.show()

# MAIN FUNCTION

def main():
    print("Starting Demand Forecasting...\n")
    df = load_data()
    data = prepare_data(df)
    model_fit = train_arima(data)
    forecast_values = forecast(model_fit, steps=30)
    plot_forecast(data, forecast_values)

    print("\n FORECAST INSIGHT:")
    print("• Predicted demand trends help optimize inventory and logistics planning.")
    print("• Businesses can reduce stockouts and overstocking using forecasts.")
    print("• Forecasting enables proactive decision-making in supply chain operations.")

    print("\nDemand Forecasting Completed!")


if __name__ == "__main__":
    main()
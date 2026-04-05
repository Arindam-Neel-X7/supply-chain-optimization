import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD DATA

def load_data():
    file_path = "../data/processed/cleaned_data.csv"
    df = pd.read_csv(file_path)
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

# BASIC INFO

def basic_info(df):
    print("\n Dataset Info:")
    print(df.info())

    print("\n Shape:", df.shape)

    print("\n Summary Statistics:")
    print(df.describe())

# SALES TREND OVER TIME

def sales_trend(df):
    # Resample to monthly for clarity
    monthly_sales = df.resample('ME', on='order_date')['sales'].sum()

    plt.figure(figsize=(12, 6))
    monthly_sales.plot()
    plt.title("Daily Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid()
    plt.show()

# TOP CITIES

def top_cities(df):
    top = df['city'].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top.values, y=top.index)
    plt.title("Top 10 Cities by Orders")
    plt.xlabel("Number of Orders")
    plt.ylabel("City")
    plt.show()

# SALES DISTRIBUTION

def sales_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['sales'], bins=50, kde=True)
    plt.title("Sales Distribution")
    plt.xlabel("Sales")
    plt.show()

# Monthly Trend

def monthly_trend(df):
    df['month'] = df['order_date'].dt.to_period('M')
    monthly_sales = df.groupby('month')['sales'].sum()

    plt.figure(figsize=(12, 6))
    monthly_sales.plot()
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.grid()
    plt.show()


# MAIN FUNCTION

def main():
    print("Starting EDA...\n")

    df = load_data()

    basic_info(df)
    sales_trend(df)
    monthly_trend(df)
    top_cities(df)
    sales_distribution(df)


    print("\n EDA Completed!")


if __name__ == "__main__":
    main()
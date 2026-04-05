import os
import pandas as pd

# Create Required Folders

def create_folders():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    print("Folders created successfully!")

file_path = "data/raw/DataCoSupplyChainDataset.csv"

# Load Dataset

def load_data(path):
    try:
        df = pd.read_csv(path, encoding='latin1')
        print("Dataset loaded successfully!")
        return df
    except Exception as e:
        print("Error loading dataset:", e)
        return None

# Basic Data Inspection

def inspect_data(df):
    print("\nACTUAL COLUMN NAMES:")
    for col in df.columns:
        print(col)

    print("\n FIRST 5 ROWS:")
    print(df.head())

    print("\n DATASET SHAPE:")
    print(df.shape)

    print("\n COLUMN NAMES:")
    print(df.columns)

    print("\n DATA INFO:")
    print(df.info())

    print("\n MISSING VALUES:")
    print(df.isnull().sum())

# Select Important Columns

def select_columns(df):
    columns = [
        'order date (dateorders)',
        'sales',
        'order item quantity',
        'order item total',
        'customer city',
        'customer country'
    ]

    df = df[columns]
    print("Selected relevant columns")
    return df

# Clean Data

def clean_data(df):
    # Rename columns
    df.columns = [
        'order_date',
        'sales',
        'quantity',
        'total_price',
        'city',
        'country'
    ]

    # Convert date
    df['order_date'] = pd.to_datetime(df['order_date'])

    # Drop missing values
    df.dropna(inplace=True)

    print("Data cleaned successfully!")
    return df

# Save Clean Data

def save_data(df):
    output_path = "data/processed/cleaned_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved at: {output_path}")

# MAIN FUNCTION (Execution Flow)

def main(path):
    print("Starting Data Collection Pipeline...\n")
    create_folders()
    df = load_data(path)
    if df is not None:
        df.columns = df.columns.str.strip().str.lower()
        inspect_data(df)
        df = select_columns(df)
        df = clean_data(df)
        save_data(df)

    print("\n Data Collection Completed!")


# ENTRY POINT

if __name__ == "__main__":
    main(file_path)
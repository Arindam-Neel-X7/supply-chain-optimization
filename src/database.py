import sqlite3
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "supply_chain.db")

# CREATE TABLE
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS live_data (
        timestamp TEXT PRIMARY KEY,
        temperature REAL,
        humidity REAL,
        trend_score REAL,
        is_holiday INTEGER,
        gdp REAL,
        price REAL,
        rating REAL,
        review_count REAL
    )
    """)

    conn.commit()
    conn.close()

# INSERT DATA
def insert_data(df):
    conn = sqlite3.connect(DB_PATH)

    for _, row in df.iterrows():
        try:
            row.to_frame().T.to_sql("live_data", conn, if_exists="append", index=False)
        except sqlite3.IntegrityError:
            print("⚠️ Duplicate timestamp skipped")

    conn.close()

# LOAD DATA
def load_db_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM live_data ORDER BY timestamp", conn)
    conn.close()
    return df
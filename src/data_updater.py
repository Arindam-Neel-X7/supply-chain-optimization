import time
from src.database import init_db, insert_data
from src.api_data_pipeline import get_real_time_features


def update_live_dataset():
    print("\n🔄 Updating database...")
    init_db()
    new_data = get_real_time_features()
    # Convert timestamp to string (IMPORTANT FIX)
    if "timestamp" in new_data.columns:
        new_data["timestamp"] = new_data["timestamp"].astype(str)
    insert_data(new_data)
    print("✅ Data stored in DB")
    # 🔥 ADD THIS (VERY IMPORTANT)
    time.sleep(5)   # wait 5 seconds between API calls

def run_data_collection(interval=300):
    while True:
        try:
            update_live_dataset()
        except Exception as e:
            print("❌ Error:", e)

        time.sleep(interval)

if __name__ == "__main__":
    run_data_collection(interval=300)
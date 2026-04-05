import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from dotenv import load_dotenv

# Load env
load_dotenv()

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_URL = f"{BASE_URL}/forecast"

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Supply Chain Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.card {
    background: linear-gradient(135deg, #1C1F26, #2A2E39);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    transition: all 0.3s ease-in-out;
}

.card:hover {
    transform: scale(1.04);
    box-shadow: 0px 6px 20px rgba(76, 175, 80, 0.4);
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
}

.metric-label {
    font-size: 16px;
    color: #AAAAAA;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Controls")

# 📅 Date Picker
start_date = st.sidebar.date_input("Start Date")

# 📦 Product Selector
product = st.sidebar.selectbox(
    "Select Product",
    ["Product A", "Product B", "Product C"]
)

# ------------------ HEADER ------------------
st.title("📦 Supply Chain Forecasting Dashboard")
st.caption(f"🔗 Connected to API: {BASE_URL}")

st.markdown("### Predict future demand using ML models 🚀")

# ------------------ BUTTON ------------------
if st.button("Generate Forecast"):

    # 🔥 Loading animation
    with st.spinner("⏳ Generating forecast..."):

        try:
            response = requests.get(API_URL, timeout=3600)

            if response.status_code == 200:
                data = response.json()
                forecast = data.get("forecast", [])

                st.success("✅ Forecast generated successfully!")
                st.write("")

                if not forecast:
                    st.warning("No forecast data received.")
                else:
                    # ------------------ DATAFRAME ------------------
                    df = pd.DataFrame({
                        "Date": pd.date_range(
                            start=start_date if start_date else pd.Timestamp.today(),
                            periods=len(forecast)
                        ),
                        "Forecast": forecast
                    })
                    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

                    def animate_value(value, prefix="", suffix=""):
                        placeholder = st.empty()
                        for i in range(int(value)):
                            placeholder.markdown(f"<h2>{prefix}{i}{suffix}</h2>", unsafe_allow_html=True)
                            time.sleep(0.01)
                        placeholder.markdown(f"<h2>{prefix}{value:.2f}{suffix}</h2>", unsafe_allow_html=True)

                    # ------------------ KPI SECTION ------------------
                    st.markdown("---")
                    st.markdown("## 📊 Performance Overview")

                    avg_demand = df["Forecast"].mean()
                    max_demand = df["Forecast"].max()
                    trend = df["Forecast"].iloc[-1] - df["Forecast"].iloc[0]

                    trend_color = "#4CAF50" if trend > 0 else "#FF4B4B"
                    arrow = "⬆️" if trend > 0 else "⬇️"

                    col1, col2, col3 = st.columns(3)

                    # 🔥 CARD 1
                    with col1:
                        st.markdown(f"""
                        <div class="card">
                           <div class="metric-label">📊 Avg Daily Demand (30D)</div>
                            <div class="metric-value" style="color:#4CAF50;">
                                {avg_demand:.2f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 🔥 CARD 2
                    with col2:
                        st.markdown(f"""
                        <div class="card">
                            <div class="metric-label">📈 Peak Demand</div>
                            <div class="metric-value" style="color:#2196F3;">
                                {max_demand:.2f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 🔥 CARD 3
                    with col3:
                        st.markdown(f"""
                        <div class="card">
                            <div class="metric-label">📉 Net Trend (Start → End)</div>
                            <div class="metric-value" style="color:{trend_color};">
                                {arrow} {trend:.2f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with st.container():

                        # KPI cards

                        st.markdown("<br>", unsafe_allow_html=True)

                        if trend > 0:
                            st.info("📈 Demand is trending upward over the next 30 days.")
                        else:
                            st.info("📉 Demand shows a declining trend over the next 30 days.")

                   # ------------------ TABLE ------------------
                    st.subheader("📋 Forecast Data")
                    st.dataframe(df, use_container_width=True)

                    csv = df.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv,
                        file_name="forecast.csv",
                        mime="text/csv",
                    )
                    # ------------------ CHART ------------------
                    st.subheader("📈 Forecast Visualization")

                    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0E1117')

                    ax.set_facecolor('#0E1117')  # 🔥 match dark theme

                    ax.plot(
                        df["Date"],
                        df["Forecast"],
                        linewidth=3,
                        color='#4CAF50',
                        marker='o',
                        markersize=4,
                        alpha=0.9
                    )
                    ax.set_xticks(df["Date"][::3])  # show every 3rd date

                    ax.set_xlabel("Date", color='white')
                    ax.set_ylabel("Demand", color='white')
                    ax.set_title(f"{product} Demand Forecast (30 Days)", color='white')

                    ax.tick_params(colors='white')

                    ax.grid(True, linestyle='--', alpha=0.3)

                    max_idx = df["Forecast"].idxmax()

                    ax.scatter(
                        df["Date"][max_idx],
                        df["Forecast"][max_idx],
                        color='red',
                        s=80,
                        label="Peak"
                    )
                    min_idx = df["Forecast"].idxmin()

                    ax.scatter(
                        df["Date"][min_idx],
                        df["Forecast"][min_idx],
                        color='blue',
                        s=80,
                        label="Lowest"
                    )

                    ax.legend(facecolor='#1C1F26', edgecolor='none', labelcolor='white')

                    fig.autofmt_xdate()

                    st.pyplot(fig)

            else:
                st.error(f"API Error: {response.status_code}")

        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Cannot connect to backend: {e}")
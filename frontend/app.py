import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
API_URL = "https://supply-chain-api-zssg.onrender.com/forecast"

st.set_page_config(
    page_title="Supply Chain Dashboard",
    layout="wide",
    page_icon="📦"
)

# ----------------------------
# CUSTOM CSS (PREMIUM UI)
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.metric-card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 0 10px rgba(0,0,0,0.4);
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
}
.metric-title {
    font-size: 14px;
    color: gray;
}
.sidebar .sidebar-content {
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR (CONTROLS)
# ----------------------------
st.sidebar.title("⚙️ Controls")

start_date = st.sidebar.date_input("Start Date")
product = st.sidebar.selectbox("Select Product", ["Product A", "Product B", "Product C"])

# ----------------------------
# HEADER
# ----------------------------
st.title("📦 Supply Chain Forecasting Dashboard")
st.markdown("🔗 Connected to API")

st.markdown("### Predict future demand using ML models 🚀")

# ----------------------------
# BUTTON
# ----------------------------
if st.button("Generate Forecast"):

    with st.spinner("Fetching forecast..."):

        response = requests.get(API_URL)
        data = response.json()

        if data["status"] == "success":
            forecast = data["forecast"]

            dates = pd.date_range(start=start_date, periods=len(forecast))

            df = pd.DataFrame({
                "Date": dates,
                "Forecast": forecast
            })

            st.success("Forecast generated successfully!")

            # ----------------------------
            # KPI CARDS
            # ----------------------------
            avg = sum(forecast) / len(forecast)
            max_val = max(forecast)
            trend = forecast[-1] - forecast[0]

            col1, col2, col3 = st.columns(3)

            col1.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Average Demand</div>
                <div class="metric-value" style="color:#4CAF50;">{avg:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            col2.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Max Demand</div>
                <div class="metric-value" style="color:#2196F3;">{max_val:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            col3.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Trend</div>
                <div class="metric-value" style="color:#FF5252;">{trend:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # ----------------------------
            # TABLE
            # ----------------------------
            st.subheader("📋 Forecast Data")
            st.dataframe(df, use_container_width=True)

            # ----------------------------
            # DOWNLOAD BUTTON
            # ----------------------------
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Forecast CSV",
                csv,
                "forecast.csv",
                "text/csv"
            )

            # ----------------------------
            # CUSTOM CHART (MATPLOTLIB)
            # ----------------------------
            st.subheader("📈 Forecast Visualization")

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(df["Date"], df["Forecast"], marker='o', color="#4CAF50")

            # Highlight peak
            peak_idx = df["Forecast"].idxmax()
            ax.scatter(df["Date"][peak_idx], df["Forecast"][peak_idx],
                       color="red", s=100, label="Peak")

            ax.set_title(f"{product} Demand Forecast (30 Days)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Demand")

            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()

            st.pyplot(fig)

        else:
            st.error("Error fetching data")
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "https://supply-chain-api-zssg.onrender.com/forecast"

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Supply Chain AI", layout="wide")

# ------------------ ULTRA CSS ------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0E1117, #111827);
}

/* Glass Cards */
.glass {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,0,0,0.5);
}

/* KPI Cards */
.metric-card {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    background: linear-gradient(145deg, #1f2937, #111827);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.03);
    box-shadow: 0 0 25px rgba(76, 175, 80, 0.7);
}

.metric-title {
    color: #9ca3af;
    font-size: 14px;
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #22c55e);
    color: white;
    border-radius: 12px;
    padding: 10px 25px;
    font-size: 16px;
    border: none;
}

/* Section Titles */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("## ⚙️ Controls")

start_date = st.sidebar.date_input("Start Date")
product = st.sidebar.selectbox("Select Product", ["Product A", "Product B", "Product C"])

# ------------------ HEADER ------------------
st.markdown("# 📦 Supply Chain AI Dashboard")
st.markdown("### 🚀 Intelligent Demand Forecasting System")

st.caption("🟢 Connected to Live API")

# ------------------ BUTTON ------------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    generate = st.button("✨ Generate Forecast")

# ------------------ MAIN ------------------
if generate:
    with st.spinner("Running AI model..."):

        response = requests.get(API_URL)
        data = response.json()

        if data["status"] == "success":
            forecast = data["forecast"]

            dates = pd.date_range(start=start_date, periods=len(forecast))

            df = pd.DataFrame({
                "Date": dates.strftime("%Y-%m-%d"),
                "Forecast": forecast
            })

            st.success("✅ Forecast generated successfully!")

            # ---------------- KPI ----------------
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
                <div class="metric-value" style="color:#60A5FA;">{max_val:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            col3.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Trend</div>
                <div class="metric-value" style="color:#F87171;">{trend:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            # ---------------- INSIGHTS ----------------
            st.markdown("### 🧠 AI Insights")

            if trend > 0:
                st.success("📈 Demand is rising — consider increasing inventory.")
            elif trend < 0:
                st.warning("📉 Demand is falling — optimize stock levels.")
            else:
                st.info("⚖️ Demand is stable.")

            # ---------------- TABLE ----------------
            st.markdown("### 📋 Forecast Data")

            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                "📥 Download Forecast",
                csv,
                "forecast.csv",
                "text/csv"
            )

            # ---------------- CHART ----------------
            st.markdown("### 📈 Forecast Visualization")

            fig, ax = plt.subplots(figsize=(12,5), facecolor='#0E1117')
            ax.set_facecolor('#0E1117')

            ax.plot(df["Date"], df["Forecast"], color="#4CAF50", marker='o')

            peak_idx = df["Forecast"].idxmax()
            ax.scatter(df["Date"][peak_idx], df["Forecast"][peak_idx],
                       color="red", s=120)

            ax.set_title(f"{product} Forecast", color='white')
            ax.tick_params(colors='white')

            for spine in ax.spines.values():
                spine.set_color('#333')

            ax.grid(alpha=0.2)

            st.pyplot(fig)

        else:
            st.error("API error")
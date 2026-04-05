import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "https://supply-chain-api-zssg.onrender.com/forecast"

st.set_page_config(page_title="Supply Chain Forecasting", layout="wide")

st.title("📦 Supply Chain Forecast Dashboard")
st.markdown("Forecasting future demand using ML 🚀")

if st.button("🔮 Generate Forecast"):

    with st.spinner("Fetching forecast..."):
        try:
            response = requests.get(API_URL)
            data = response.json()

            if data["status"] == "success":
                forecast = data["forecast"]

                df = pd.DataFrame({
                    "Day": range(1, len(forecast)+1),
                    "Forecast": forecast
                })

                st.success("Forecast generated successfully!")

                st.line_chart(df.set_index("Day"))
                st.dataframe(df)

            else:
                st.error(data["message"])

        except Exception as e:
            st.error(f"Error: {str(e)}")
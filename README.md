🚀 📦 AI-Powered Supply Chain Forecasting System
⚡ A production-grade, full-stack machine learning system for real-time demand forecasting with an interactive analytics dashboard.

🌐 Live Demo :

🔗 Frontend (Dashboard): https://supply-chain-optimization-x7.streamlit.app

🔗 Backend API : https://supply-chain-api-zssg.onrender.com

🧠 💡 Why This Project?
Supply chain inefficiencies cost companies billions annually due to:
overstocking 📦
understocking 📉
inaccurate demand predictions

🎯 ✨ Key Highlights

🔥 Full-stack ML system (Model + API + UI)
🔥 Real-time forecasting pipeline
🔥 Hybrid intelligent modeling (ARIMA + fallback logic)
🔥 Interactive dashboard with business insights
🔥 Production-ready architecture (model caching, API design)

🏗️ ⚙️ System Architecture : 
        ┌────────────────────┐
        │  Streamlit UI      │
        │  (Interactive App) │
        └─────────┬──────────┘
                  ↓
        ┌────────────────────┐
        │   FastAPI Backend  │
        │   (Render Cloud)   │
        └─────────┬──────────┘
                  ↓
        ┌────────────────────┐
        │  ML Engine         │
        │  (ARIMA + Hybrid)  │
        └─────────┬──────────┘
                  ↓
        ┌────────────────────┐
        │ Forecast + Insights│
        └────────────────────┘

🤖 📊 Machine Learning Strategy
🔹 Core Model
   Time-series forecasting using ARIMA
🔹 Intelligent Hybrid System

Automatically selects best strategy based on data quality:
Data Condition	          Model Used
High variation	            ARIMA
Low variation	            Smoothed ARIMA
Sparse data	            Mean-based fallback

🔹 Real-Time Enhancement
    Controlled stochastic variation for “live” forecasting
    Negative value correction
    Trend-aware predictions

📊 📈 Dashboard Features

✨ Interactive Plotly charts
📊 KPI Metrics (Average, Max, Trend)
🔴 Peak demand detection
🔵 Lowest demand highlighting
🧠 AI-generated insights
📅 Clean date visualization
📥 Downloadable CSV reports

🎨 🖥️ UI Preview
    A modern, dark-themed SaaS-style dashboard built with Streamlit
    Glassmorphism UI
    Hover effects & animations
    Interactive analytics
    Clean UX

⚡ 🚀 Performance Optimization
Feature	                    Impact
Model caching	          ⚡ Instant response
Startup training	      ❌ No retraining per request
Optimized ARIMA	            Faster inference
Lightweight pipeline	      Scalable

🛠️ 🧰 Tech Stack
🔹 Backend
    Python
    FastAPI
    Statsmodels (ARIMA)
    Pandas / NumPy

🔹 Frontend
    Streamlit
    Plotly

🔹 Deployment
    Render (Backend API)
    Streamlit Cloud (Frontend)
    Git & GitHub

📦 📂 Project Structure :
supply-chain-optimization/
│
├── src/
│   ├── advanced_forecasting.py
│
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│
├── data/
│   ├── processed/
│
├── main.py
├── requirements.txt

🚀 ⚙️ How to Run Locally:
1️⃣ Clone the repository:
git clone https://github.com/your-username/your-repo.git
cd your-repo

2️⃣ Run Backend:
pip install -r requirements.txt
uvicorn main:app --reload

3️⃣ Run Frontend:
cd frontend
streamlit run app.py

📊 📌 Example Output

✔ 30-day demand forecast
✔ Interactive chart with trends
✔ Business insights (increase/decrease demand)

📊 Dashboard
![Dashboard](assets/Dashboard.png)

Forecast Table
![Forecast Table](assets/Forecast_Table.png)

📈 Forecast Chart
![Chart](assets/Forecast_chart.png)

🔮 🚀 Future Enhancements
    📊 Multi-product comparison dashboard
    🗄 Database integration (PostgreSQL)
    🔄 Automated retraining pipeline
    🔐 User authentication
    🌐 Custom domain deployment
    📈 Advanced models (Prophet, LSTM)

👨‍💻 👤 Author
Arindam Karmakar
GitHub : https://github.com/Arindam-Neel-X7
Linkedin : https://www.linkedin.com/in/arindamkarmakarx

⭐ Support
If you found this project useful:
Give it a ⭐ on GitHub
Share it with others

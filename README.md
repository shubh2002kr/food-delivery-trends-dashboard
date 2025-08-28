# 🍽️ Food Delivery Trends — Streamlit App  

[![Streamlit](https://img.shields.io/badge/Made%20With-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)  
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)  
[![Plotly](https://img.shields.io/badge/Charts-Plotly-3DDC84?logo=plotly&logoColor=white)](https://plotly.com/python/)  
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)  
[![NumPy](https://img.shields.io/badge/Data-NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)  
[![Statsmodels](https://img.shields.io/badge/Analysis-Statsmodels-008000?logo=python&logoColor=white)](https://www.statsmodels.org/)  

An **interactive dashboard** to analyze **Zomato & Swiggy order data** with insights on **delivery times, ratings, revenue, cuisines, and demand patterns**.  

---

## 🌐 Live Demo  
👉 [Click here to use the app](https://your-streamlit-deploy-link.streamlit.app/)  

---

## ✨ Features
- 📂 **Drag & Drop CSVs** (Zomato, Swiggy, or combined files)  
- 📊 **KPIs** → Total Orders, Avg Rating, Avg Delivery Time, Revenue, Delay Rate  
- 📈 **Visuals** → Orders & Revenue over time, Delivery & Ratings distribution  
- 🍔 **Top Entities** → Popular cuisines and restaurants  
- ⏰ **Demand Patterns** → Peak hours, weekday trends  
- 🌍 **Geo Insights** → Order locations (if lat/lon available)  
- 🏢 **Platform & City Breakdown** → Compare Zomato vs Swiggy across cities  
- ⬇️ Export filtered dataset as CSV  

---

## ⚡ Run Locally
If you want to run the app on your machine:  
```bash
pip install -r requirements.txt
streamlit run app.py

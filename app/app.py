import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from recommender import generate_recommendations
from utils.pipeline import ingest_batch

st.set_page_config(page_title="SmartChain", layout="wide")

# Load model and scaler
model = joblib.load("models/inspection_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load data from DB
@st.cache_data
def load_data():
    conn = sqlite3.connect("db/supplychain.db")
    df = pd.read_sql("SELECT * FROM supply_data", conn)
    return df

# Simulate ingestion
if st.sidebar.button("Ingest New Batch"):
    source_df = pd.read_csv("data/supply_chain_data.csv")
    ingest_batch(source_df)
    st.sidebar.success("New data ingested successfully!")

df = load_data()

st.title("📦 SmartChain: Inspection Prediction & Optimization")

tab1, tab2, tab3 = st.tabs(["📊 Insights", "⚙️ Predict", "📌 Recommendations"])

with tab1:
    st.subheader("Feature Correlations")
    st.write(df.corr(numeric_only=True)["Inspection results"].sort_values())

    # Business Insights
    st.markdown("🚨 **Insight:** Shipping time has a strong negative correlation (-0.65) with inspection results. Consider auditing routes or choosing faster carriers.")
    st.markdown("📦 **Insight:** Orders with smaller quantities tend to pass inspection more often. Suggest breaking large orders into batches.")
    st.markdown("🔄 **Insight:** Long supplier lead times negatively impact inspection outcomes. Negotiate better SLAs or choose alternate suppliers.")
    st.markdown("📈 **Insight:** High availability and strong internal lead time management contribute positively to inspection success.")

    st.subheader("Cost Breakdown")
    avg_cost = df.groupby("Inspection results")["Manufacturing costs"].mean()
    st.bar_chart(avg_cost)

    st.subheader("Optimal Range Stats")
    filtered = df[(df["Defect rates"] < 6) & (df["Availability"] > 70) & (df["Shipping costs"] < 400)]
    pass_rate = filtered["Inspection results"].mean() * 100
    st.markdown(f"✅ Products with low defect (<6%), high availability (>70%) and shipping cost < ₹400 have a **{pass_rate:.2f}%** pass rate.")

    # Natural language summary
    st.markdown("""
    ### 🔍 Summary:
    - **Shipping times** are the most critical bottleneck—strongly linked to inspection failures.
    - **Order quantities** and **lead times** have moderate negative impact—monitor large orders and slow suppliers.
    - **Availability** and **revenue** are weak but positive indicators of inspection success.
    """)

with tab2:
    st.subheader("Enter Features for Prediction")

    inputs = {
        "Price": st.number_input("Price", 0.0, 1000.0, 100.0),
        "Availability": st.number_input("Availability", 0, 500, 100),
        "Stock levels": st.number_input("Stock levels", 0, 500, 150),
        "Lead times": st.number_input("Lead times", 0, 50, 10),
        "Order quantities": st.number_input("Order quantities", 0, 1000, 200),
        "Shipping times": st.number_input("Shipping times", 0, 30, 5),
        "Shipping costs": st.number_input("Shipping costs", 0.0, 1000.0, 100.0),
        "Lead time": st.number_input("Lead time", 0, 50, 10),
        "Production volumes": st.number_input("Production volumes", 0, 1000, 300),
        "Manufacturing lead time": st.number_input("Manufacturing lead time", 0, 60, 20),
        "Manufacturing costs": st.number_input("Manufacturing costs", 0.0, 1000.0, 400.0),
        "Defect rates": st.number_input("Defect rates", 0.0, 100.0, 5.0),
        "Transportation modes": st.selectbox("Transportation Mode (encoded)", [0, 1, 2]),
        "Routes": st.selectbox("Route (encoded)", [0, 1, 2, 3])
    }

    input_df = pd.DataFrame([inputs])
    scaled_input = scaler.transform(input_df)
    pred = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]
    label = "✅ Pass" if pred == 1 else "❌ Fail"
    st.subheader("Prediction Result:")
    st.markdown(f"### {label}")
    st.markdown(f"**Pass Probability:** {proba*100:.2f}%")

with tab3:
    st.subheader("Business Recommendations")
    recs = generate_recommendations(df)
    if recs:
        for r in recs:
            st.markdown(r)
    else:
        st.success("✅ No major supply chain risks found in current batch.") 
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Set page config
st.set_page_config(page_title="Customer CLV Prediction", layout="centered")
st.title("🔮 Predict Customer Value Using RFM Features")

# ✅ Load model (no Path, no pickle, just joblib)
@st.cache_resource
def load_model():
    return joblib.load("xgb_reg_model.pkl")

# Load once
model = load_model()

# ---- INPUT FORM ----
with st.form("input_form"):
    st.subheader("Enter a Single Transaction")

    customer_id = st.number_input("Customer ID", min_value=1, step=1)
    date_input = st.date_input("Transaction Date", value=datetime(1997, 1, 1).date())
    quantity = st.number_input("Quantity", min_value=1, step=1)
    price = st.number_input("Price per Unit", min_value=0.0, step=0.1)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Step 1: Create transaction DataFrame
        df = pd.DataFrame([{
            "customer_id": customer_id,
            "date": pd.to_datetime(date_input),
            "quantity": quantity,
            "price": price
        }])
        df["total_price"] = df["quantity"] * df["price"]

        # Step 2: Reference date
        reference_date = pd.to_datetime("1998-01-01")
        recency = (reference_date - df["date"].max()).days

        # Step 3: Compute RFM-like features
        rfm_features = pd.DataFrame([{
            "recency": recency,
            "frequency": 1,  # one transaction
            "price_sum": df["total_price"].sum(),
            "price_mean": df["price"].mean()
        }])

        # Step 4: Predict
        prediction = model.predict(rfm_features)[0]

        st.success(f"🧾 Predicted Customer Value: ₹{prediction:.2f}")
        st.subheader("📄 Model Input Features")
        st.write(rfm_features)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")


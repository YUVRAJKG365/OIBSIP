# app.py

import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    try:
        with open("car_price_prediction_model.pkl", "rb") as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        st.error("🚨 Model file `car_price_prediction_model.pkl` not found!")
        return None


@st.cache_data
def load_data():
    try:
        return pd.read_csv("car data.csv")
    except FileNotFoundError:
        st.error("🚨 Data file `car data.csv` not found!")
        return None


model = load_model()
df_original = load_data()

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(120deg, #f0f4f9, #e0f2fe);
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #1e3a8a;
        font-weight: 700;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #1e3a8a, #2563eb);
        color: white;
        border-radius: 40px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        transform: scale(1.05);
    }

    /* Number Input, Selectbox, Text Input */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px;
        border: 2px solid #93c5fd;
        padding: 6px;
        background-color: #f9fafb;
    }

    /* Prediction Result */
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    }
    .success {
        background: #16a34a;
        color: white;
    }
    .error {
        background: #dc2626;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- MAIN APP --------------------
def main():
    st.markdown("<h1 style='text-align:center;'>🚗 Smart Car Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#6b7280;'>by Yuvraj Kumar Gond</h3>", unsafe_allow_html=True)
    st.write("<p style='text-align:center; color:#374151;'>Estimate the fair selling price of your used car with AI precision ⚡</p>", unsafe_allow_html=True)
    st.markdown("---")

    if model is None or df_original is None:
        st.stop()

    # --- User Input Section ---
    st.header("🔧 Enter Car Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        car_name = st.selectbox(
            "Car Brand & Model",
            options=sorted(df_original["Car_Name"].unique())
        )
        year = st.number_input(
            "Manufacturing Year",
            min_value=2000,
            max_value=datetime.now().year,
            value=2015,
            help="Enter the year the car was manufactured."
        )
        present_price = st.number_input(
            "Current Showroom Price (Lakhs)",
            min_value=0.1,
            value=5.5,
            step=0.1,
            help="Price of the car if bought new today."
        )

    with col2:
        kms_driven = st.number_input("Kilometers Driven", min_value=100, value=30000, step=100)
        fuel_type = st.selectbox("Fuel Type", options=df_original["Fuel_Type"].unique())
        selling_type = st.selectbox("Selling Type", options=df_original["Selling_type"].unique())

    with col3:
        transmission = st.selectbox("Transmission", options=df_original["Transmission"].unique())
        owner = st.number_input("Number of Previous Owners", min_value=0, max_value=10, value=0)

    # --- Prediction ---
    if st.button("🔮 Predict Price", use_container_width=True):
        with st.spinner("Calculating best price... ⏳"):

            # Get model training columns
            categorical_cols = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
            df_encoded = pd.get_dummies(df_original, columns=categorical_cols, drop_first=True)
            df_encoded["Car_Age"] = datetime.now().year - df_encoded["Year"]
            model_columns = df_encoded.drop(columns=["Selling_Price", "Car_Name_ritz", "Year"]).columns

            # Create input dataframe
            input_dict = {
                "Car_Name": car_name,
                "Year": year,
                "Present_Price": present_price,
                "Driven_kms": kms_driven,
                "Fuel_Type": fuel_type,
                "Selling_type": selling_type,
                "Transmission": transmission,
                "Owner": owner,
            }
            input_df = pd.DataFrame([input_dict])

            input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
            input_encoded["Car_Age"] = datetime.now().year - input_encoded["Year"]
            final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

            prediction = model.predict(final_input)

        st.markdown(
            f'<div class="result-box success">💰 Predicted Selling Price: <br><span style="font-size:1.5rem;">₹ {prediction[0]:.2f} Lakhs</span></div>',
            unsafe_allow_html=True
        )
        st.balloons()

    st.markdown("---")

    # --- Sidebar ---
    st.sidebar.header("📘 About the Project")
    st.sidebar.info(
        "This app predicts the selling price of a used car using **Machine Learning**. "
        "It considers factors like brand, fuel type, transmission, ownership, and kilometers driven."
    )
    st.sidebar.header("🧠 Model Details")
    st.sidebar.info(
        "Model: **Linear Regression**\n\n"
        "Dataset: 301 records of car sales\n\n"
        "Goal: Provide quick, reliable, and interpretable car price estimates."
    )


if __name__ == "__main__":
    main()

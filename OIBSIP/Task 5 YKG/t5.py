import streamlit as st
import pandas as pd
import joblib
import os

# --- Set your trained model path here ---
MODEL_PATH = r"C:\\Users\\yuvra\\Documents\\OIBSIP\\Task 5 YKG\\linear_regression_model.pkl"  # Change this to your actual model file

st.set_page_config(page_title="Sales Prediction", page_icon="📈", layout="centered")
st.title('📈 Sales Prediction for Advertising')

# Try to load the model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error(f"Model file not found at: {MODEL_PATH}")

if model is None:
    st.info("Please check the model path and file.")
else:
    st.header("Advertising Budget")
    tv_spend = st.slider('TV Advertising Spend ($)', min_value=0, max_value=300, value=150, step=1)
    radio_spend = st.slider('Radio Advertising Spend ($)', min_value=0, max_value=50, value=25, step=1)
    newspaper_spend = st.slider('Newspaper Advertising Spend ($)', min_value=0, max_value=120, value=30, step=1)

    if st.button('Predict Sales', type="primary"):
        input_data = pd.DataFrame({
            'TV': [tv_spend],
            'Radio': [radio_spend],
            'Newspaper': [newspaper_spend]
        })
        prediction = model.predict(input_data)
        predicted_sales = prediction[0]
        st.success(f'Predicted Sales: **${predicted_sales:,.2f}k**')

st.markdown(
    """
    ---
    *Built with Python and Streamlit, based on a Linear Regression model.*
    """
)

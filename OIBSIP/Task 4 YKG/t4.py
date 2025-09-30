import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
import joblib
st.set_page_config(page_title="Smart Email Spam Detector", page_icon="📧", layout="wide")

# ---------------- PREPROCESSING FUNCTION ----------------
def clean_text(text):
    """Cleans the input text by converting to lowercase, removing punctuation,
    and removing stopwords (same process as model training)."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()

    try:
        stop_words = stopwords.words('english')
    except LookupError:
        st.info("Downloading necessary NLTK data (stopwords)...")
        nltk.download('stopwords')
        stop_words = stopwords.words('english')
        st.success("Download complete. Please refresh if the app doesn't update.")

    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


# ---------------- LOAD MODEL & VECTORIZER ----------------
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load("spam_detector_model.joblib")
        vectorizer = joblib.load("tfidf_vectorizer.joblib")
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model or vectorizer not found! Please keep "
                 "`spam_detector_model.joblib` & `tfidf_vectorizer.joblib` "
                 "in the same folder as this script.")
        return None, None


model, vectorizer = load_model_and_vectorizer()


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    /* App Background */
    .stApp {
        background: linear-gradient(135deg, #f0f4f9, #dbeafe);
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #1e3a8a;
        font-weight: 700;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #1e3a8a, #2563eb);
        color: white;
        border-radius: 40px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        transform: scale(1.05);
    }

    /* Text Area */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #93c5fd;
        background-color: #f9fafb;
        padding: 10px;
    }

    /* Result Cards */
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 1rem;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .spam {
        background: #dc2626;
        color: white;
    }
    .ham {
        background: #16a34a;
        color: white;
    }

    /* Tabs Padding */
    .stTabs [role="tablist"] {
        gap: 2rem;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- APP TITLE ----------------
st.markdown("<h1 style='text-align:center;'>📧 Smart Email Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#4b5563;'>by Yuvraj Kumart Gond</h3>", unsafe_allow_html=True)
st.write("<p style='text-align:center; color:#374151;'>An AI-powered app that classifies emails as <b>Spam</b> or <b>Ham</b> in seconds 🚀</p>", unsafe_allow_html=True)


# ---------------- MAIN APP ----------------
if model and vectorizer:
    tab1, tab2 = st.tabs(["✍️ Write a Message", "📂 Upload a File"])

    # ---- Tab 1: Write Message ----
    with tab1:
        st.subheader("🔍 Analyze a Typed Message")
        message_input = st.text_area("Enter the email content below:", height=180, placeholder="Type or paste your email...")

        col1, col2 = st.columns([1, 2])
        with col1:
            check_button = st.button("Analyze Message")

        if check_button and message_input:
            with st.spinner("🔎 Analyzing your email..."):
                cleaned_message = clean_text(message_input)
                message_vector = vectorizer.transform([cleaned_message])
                prediction = model.predict(message_vector)[0]

            st.write("---")
            st.subheader("📌 Analysis Result")
            if prediction == 'spam':
                st.markdown('<div class="result-box spam">🚨 This looks like <b>SPAM</b>!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box ham">✅ This looks like <b>HAM</b> (Not Spam).</div>', unsafe_allow_html=True)


    # ---- Tab 2: Upload File ----
    with tab2:
        st.subheader("📂 Upload & Check an Email File")
        uploaded_file = st.file_uploader("Upload a text file (.txt or .eml):", type=['txt', 'eml'])

        if uploaded_file is not None:
            try:
                file_content = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # reset cursor
                file_content = uploaded_file.read().decode("latin-1")

            st.text_area("📜 File Content Preview:", file_content[:2000], height=200, disabled=True)

            with st.spinner("Processing file..."):
                cleaned_content = clean_text(file_content)
                content_vector = vectorizer.transform([cleaned_content])
                prediction = model.predict(content_vector)[0]

            st.write("---")
            st.subheader("📌 Analysis Result")
            if prediction == 'spam':
                st.markdown('<div class="result-box spam">🚨 This file contains <b>SPAM</b>!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box ham">✅ This file contains <b>HAM</b> (Not Spam).</div>', unsafe_allow_html=True)

else:
    st.warning("⚠️ Application not ready. Please check the error above.")

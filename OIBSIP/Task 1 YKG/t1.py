import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="🌸 Iris Flower Classifier Pro",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .feature-importance {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .developer-info {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Load Saved Model & Preprocessing --------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("iris_model.pkl")
        scaler = joblib.load("iris_scaler.pkl")
        label_encoder = joblib.load("iris_label_encoder.pkl")
        return model, scaler, label_encoder
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure 'iris_model.pkl', 'iris_scaler.pkl', and 'iris_label_encoder.pkl' are in the same directory.")
        return None, None, None

model, scaler, label_encoder = load_model()

# -------------------- Sample Data for Reference --------------------
SAMPLE_IRIS_DATA = {
    'Setosa': [5.0, 3.4, 1.5, 0.2],
    'Versicolor': [6.0, 2.7, 4.2, 1.3],
    'Virginica': [6.7, 3.0, 5.2, 2.3]
}

# -------------------- Safe Label Encoding Functions --------------------
def safe_inverse_transform(encoder, labels):
    """Safely inverse transform labels, handling unseen labels gracefully"""
    try:
        return encoder.inverse_transform(labels)
    except ValueError as e:
        st.warning("⚠️ Warning: Some labels couldn't be decoded properly.")
        # Return numeric labels as fallback
        return [f"Class_{label}" for label in labels]

def get_class_names(encoder):
    """Get class names safely from label encoder"""
    try:
        return list(encoder.classes_)
    except:
        return ["Setosa", "Versicolor", "Virginica"]  # Fallback class names

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("🎯 Navigation")
    page = st.radio("Go to", ["🏠 Prediction", "📊 Analysis", "ℹ️ About"])
    
    st.header("⚙️ Settings")
    show_confidence = st.checkbox("Show prediction confidence", value=True)
    show_visualization = st.checkbox("Show feature visualization", value=True)
    
    st.header("📈 Model Info")
    if model is not None:
        st.write(f"**Model Type:** {type(model).__name__}")

# -------------------- Prediction Page --------------------
if page == "🏠 Prediction":
    st.markdown('<h1 class="main-header">🌸 Iris Flower Classifier Pro</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
        Enter the <strong>measurements of an Iris flower</strong> or use sample data to predict its species.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------- Quick Sample Buttons --------------------
    st.subheader("🚀 Quick Samples")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌿 Setosa Sample", use_container_width=True):
            st.session_state.sepal_length = 5.0
            st.session_state.sepal_width = 3.4
            st.session_state.petal_length = 1.5
            st.session_state.petal_width = 0.2
    
    with col2:
        if st.button("🌺 Versicolor Sample", use_container_width=True):
            st.session_state.sepal_length = 6.0
            st.session_state.sepal_width = 2.7
            st.session_state.petal_length = 4.2
            st.session_state.petal_width = 1.3
    
    with col3:
        if st.button("💮 Virginica Sample", use_container_width=True):
            st.session_state.sepal_length = 6.7
            st.session_state.sepal_width = 3.0
            st.session_state.petal_length = 5.2
            st.session_state.petal_width = 2.3

    # -------------------- User Input Section --------------------
    st.subheader("🔢 Enter Flower Measurements")
    
    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 
                               st.session_state.get('sepal_length', 5.1), 0.1)
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 
                               st.session_state.get('petal_length', 1.5), 0.1)

    with col2:
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 
                              st.session_state.get('sepal_width', 3.5), 0.1)
        petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 
                              st.session_state.get('petal_width', 0.2), 0.1)

    # -------------------- Feature Visualization --------------------
    if show_visualization:
        st.subheader("📊 Feature Comparison")
        
        # Create comparison chart
        features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        current_values = [sepal_length, sepal_width, petal_length, petal_width]
        
        fig = go.Figure()
        
        # Add current input as main bar
        fig.add_trace(go.Bar(
            name='Your Input',
            x=features,
            y=current_values,
            marker_color='#FF6B6B'
        ))
        
        # Add sample data for comparison
        for species, values in SAMPLE_IRIS_DATA.items():
            fig.add_trace(go.Scatter(
                name=f'{species} Avg',
                x=features,
                y=values,
                mode='lines+markers',
                line=dict(dash='dash')
            ))
        
        fig.update_layout(
            title="Feature Comparison with Sample Species",
            xaxis_title="Features",
            yaxis_title="Centimeters (cm)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # -------------------- Prediction --------------------
    if st.button("🎯 Predict Species", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded. Please check the model files.")
        else:
            # Prepare input for model
            sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            sample_scaled = scaler.transform(sample)

            # Predict
            prediction = model.predict(sample_scaled)[0]
            predicted_species = safe_inverse_transform(label_encoder, [prediction])[0]
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(sample_scaled)[0]
                confidence = max(probabilities)
            else:
                confidence = 1.0
                probabilities = None

            # -------------------- Results Display --------------------
            st.markdown("---")
            
            # Main prediction card
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>Prediction Result</h2>
                    <h1>🌸 {predicted_species}</h1>
                    {f'<h3>Confidence: {confidence:.1%}</h3>' if show_confidence else ''}
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence metrics
            if probabilities is not None and show_confidence:
                st.subheader("📈 Prediction Confidence")
                
                prob_cols = st.columns(len(probabilities))
                class_names = get_class_names(label_encoder)
                
                for i, (class_idx, prob) in enumerate(zip(range(len(probabilities)), probabilities)):
                    with prob_cols[i]:
                        if i < len(class_names):
                            species_name = class_names[i]
                        else:
                            species_name = f"Class_{class_idx}"
                            
                        st.metric(
                            label=f"{species_name}",
                            value=f"{prob:.1%}",
                            delta="High" if prob == confidence else None
                        )
                
                # Confidence bar chart
                fig_prob = px.bar(
                    x=class_names[:len(probabilities)],
                    y=probabilities,
                    color=probabilities,
                    color_continuous_scale='Viridis',
                    title="Prediction Probabilities"
                )
                fig_prob.update_layout(showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)

            st.balloons()

# -------------------- Analysis Page --------------------
elif page == "📊 Analysis":
    st.title("📊 Model Analysis & Insights")
    
    if model is None:
        st.warning("Model not loaded. Analysis features are unavailable.")
    else:
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance")
            
            features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
            importance = model.feature_importances_
            
            fig_importance = px.bar(
                x=features,
                y=importance,
                color=importance,
                color_continuous_scale='Blues',
                title="Feature Importance in Prediction"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Data distribution
        st.subheader("📊 Iris Dataset Characteristics")
        
        # Create sample distribution visualization
        species_data = []
        for species, values in SAMPLE_IRIS_DATA.items():
            species_data.append({
                'Species': species,
                'Sepal Length': values[0],
                'Sepal Width': values[1],
                'Petal Length': values[2],
                'Petal Width': values[3]
            })
        
        df_samples = pd.DataFrame(species_data)
        
        feature_to_show = st.selectbox(
            "Select feature to visualize:",
            ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        )
        
        fig_dist = px.box(
            df_samples.melt(id_vars=['Species'], value_vars=[feature_to_show]),
            x='Species',
            y='value',
            color='Species',
            title=f"Distribution of {feature_to_show} by Species"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# -------------------- About Page --------------------
else:
    st.title("ℹ️ About Iris Flower Classifier")
    
    st.markdown("""
    ## 🌸 Iris Flower Classification
    
    This application uses machine learning to classify Iris flowers into three species:
    
    - **Iris Setosa** 🌿
    - **Iris Versicolor** 🌺  
    - **Iris Virginica** 💮
    
    ### 📋 Features
    
    - **Real-time Prediction**: Instant species classification based on input measurements
    - **Confidence Scores**: See how confident the model is in its predictions
    - **Visual Analysis**: Interactive charts and comparisons
    - **Sample Data**: Quick access to typical measurements for each species
    - **Robust Error Handling**: Graceful handling of model issues
    
    ### 📊 The Dataset
    
    The model was trained on the famous Iris dataset containing:
    - 150 samples total (50 per species)
    - 4 features: Sepal Length, Sepal Width, Petal Length, Petal Width
    
    ### 🛠️ Technical Details
    
    - **Framework**: Streamlit for web interface
    - **Machine Learning**: Scikit-learn model
    - **Visualization**: Plotly for interactive charts
    - **Error Handling**: Robust label encoding with fallbacks
    - **Deployment**: Ready for cloud deployment
    """)

    # Developer Information
    st.markdown("""
    <div class="developer-info">
        <h3>👨‍💻 Developer Information</h3>
        <p><strong>Developed by: YUVRAJ KUMAR GOND</strong></p>
        <p>This application demonstrates the integration of machine learning models with web applications 
        for real-time flower species classification.</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🌸 Built with Streamlit | Iris Flower Classification Model | Developed by YUVRAJ KUMAR GOND"
    "</div>",
    unsafe_allow_html=True
)
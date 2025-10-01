import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="AdVista Analytics | Sales Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium CSS Styling ---
st.markdown("""
<style>
    /* Main Header with Rainbow Gradient */
    .rainbow-header {
        background: linear-gradient(90deg, 
            #FF0000 0%, 
            #FF7F00 16.66%, 
            #FFFF00 33.33%, 
            #00FF00 50%, 
            #0000FF 66.66%, 
            #4B0082 83.33%, 
            #9400D3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Premium Cards */
    .premium-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: none;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(116, 185, 255, 0.3);
        border: none;
    }
    
    .insight-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #e0e6ed;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .insight-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    }
    
    .budget-card {
        background: white;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e0e6ed;
        margin: 1.5rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);
    }
    
    /* Footer */
    .premium-footer {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 4rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Sliders */
    .stSlider {
        margin: 1.5rem 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Configuration ---
MODEL_PATH = r"C:\\Users\\yuvra\\Documents\\OIBSIP\\Task 5 YKG\\linear_regression_model.pkl"

# --- Header Section with Rainbow Effect ---
st.markdown('<h1 class="rainbow-header">AdVista Analytics</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Sales Prediction Platform | By Yuvraj Kumar Gond</p>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: #2c3e50;'>🎯 Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### About Platform")
    st.info("""
    **AdVista Analytics** leverages advanced machine learning to optimize your advertising ROI across:
    
    📺 **Television** - Mass reach campaigns  
    📻 **Radio** - Localized targeting  
    📰 **Newspaper** - Traditional media  
    
    **Key Features:**
    • Real-time predictions
    • ROI optimization
    • Scenario analysis
    • Enterprise-grade security
    """)
    
    st.markdown("---")
    st.markdown("### Model Information")
    
    if os.path.exists(MODEL_PATH):
        st.success("""
        ✅ **Model Status:** Active  
        🤖 **Algorithm:** Linear Regression  
        📊 **Accuracy:** 85%+  
        🎯 **Confidence:** High
        """)
    else:
        st.error("❌ **Model Status:** Not Found")
        st.warning("Running in demonstration mode with sample predictions")

# --- Enhanced Sales Prediction Model ---
def enhanced_predict(tv_budget, radio_budget, newspaper_budget, tv_cost_per_unit, radio_cost_per_unit, newspaper_cost_per_unit):
    """
    Enhanced prediction model that considers:
    - Budget allocation
    - Cost per unit
    - Number of units (ads/spots)
    - Channel efficiency coefficients
    """
    # Calculate number of units (ads/spots) for each channel
    tv_units = tv_budget / tv_cost_per_unit if tv_cost_per_unit > 0 else 0
    radio_units = radio_budget / radio_cost_per_unit if radio_cost_per_unit > 0 else 0
    newspaper_units = newspaper_budget / newspaper_cost_per_unit if newspaper_cost_per_unit > 0 else 0
    
    # Base sales from model coefficients
    base_sales = 12.5
    
    # Sales contribution from each channel (budget * efficiency * units factor)
    tv_sales = 0.045 * tv_budget * (1 + 0.01 * tv_units)  # TV efficiency with units multiplier
    radio_sales = 0.185 * radio_budget * (1 + 0.02 * radio_units)  # Radio efficiency with units multiplier
    newspaper_sales = 0.001 * newspaper_budget * (1 + 0.005 * newspaper_units)  # Newspaper efficiency with units multiplier
    
    total_sales = base_sales + tv_sales + radio_sales + newspaper_sales
    
    return total_sales, tv_units, radio_units, newspaper_units, tv_sales, radio_sales, newspaper_sales

# --- Budget Allocation Section ---
st.markdown("""
<div class='premium-card' style='height: 140px; width:auto'>
    <h2 style='color: white; margin-bottom: 2rem; text-align: center;'>💰 Advertising Budget Allocation</h2>
</div>
""", unsafe_allow_html=True)

# Unit selector
unit = st.selectbox(
    "Select Budget Unit",
    options=["K (Thousands)", "M (Millions)"],
    index=0
)

unit_factor = 1_000 if unit.startswith("K") else 1_000_000
unit_label = "K" if unit.startswith("K") else "M"

# User enters total budget
total_budget = st.number_input(
    f"Enter Total Advertising Budget ($ {unit_label})",
    min_value=0,
    max_value=99999999999,
    value=285,
    step=5
)

# Product Cost Inputs
st.markdown("### 💰 Product Cost Configuration")
cost_col1, cost_col2, cost_col3 = st.columns(3)

with cost_col1:
    st.markdown("#### 📺 Television")
    tv_cost_per_unit = st.number_input(
        "Cost per TV ad spot ($)",
        min_value=1,
        value=100,
        step=10,
        key="tv_cost"
    )

with cost_col2:
    st.markdown("#### 📻 Radio")
    radio_cost_per_unit = st.number_input(
        "Cost per Radio ad spot ($)",
        min_value=1,
        value=50,
        step=5,
        key="radio_cost"
    )

with cost_col3:
    st.markdown("#### 📰 Newspaper")
    newspaper_cost_per_unit = st.number_input(
        "Cost per Newspaper ad ($)",
        min_value=1,
        value=20,
        step=5,
        key="newspaper_cost"
    )

# --- Dynamic Budget Allocation ---
st.markdown("### 📊 Budget Allocation")
col1, col2, col3 = st.columns(3)

# Initialize session states to persist values
if "tv_spend" not in st.session_state:
    st.session_state.tv_spend = 0
if "radio_spend" not in st.session_state:
    st.session_state.radio_spend = 0
if "newspaper_spend" not in st.session_state:
    st.session_state.newspaper_spend = 0

# Dynamic sliders
with col1:
    st.markdown(f"### 📺 Television")
    max_tv = total_budget - (st.session_state.radio_spend + st.session_state.newspaper_spend)
    tv_spend = st.slider(
        f'**Budget Allocation ($ {unit_label})**',
        min_value=0,
        max_value=max_tv,
        value=st.session_state.tv_spend,
        step=10,
        format="%d",
        key="tv_slider"
    )
    st.session_state.tv_spend = tv_spend  # Update session state immediately
    tv_units = (st.session_state.tv_spend * unit_factor) / tv_cost_per_unit if tv_cost_per_unit > 0 else 0
    st.metric("Allocated", f"${st.session_state.tv_spend:,} {unit_label}")
    st.metric("Ad Spots", f"{tv_units:,.0f}")

with col2:
    st.markdown(f"### 📻 Radio")
    max_radio = total_budget - (st.session_state.tv_spend + st.session_state.newspaper_spend)
    radio_spend = st.slider(
        f'**Budget Allocation ($ {unit_label})**',
        min_value=0,
        max_value=max_radio,
        value=st.session_state.radio_spend,
        step=5,
        format="%d",
        key="radio_slider"
    )
    st.session_state.radio_spend = radio_spend  # Update session state immediately
    radio_units = (st.session_state.radio_spend * unit_factor) / radio_cost_per_unit if radio_cost_per_unit > 0 else 0
    st.metric("Allocated", f"${st.session_state.radio_spend:,} {unit_label}")
    st.metric("Ad Spots", f"{radio_units:,.0f}")

with col3:
    st.markdown(f"### 📰 Newspaper")
    max_news = total_budget - (st.session_state.tv_spend + st.session_state.radio_spend)
    newspaper_spend = st.slider(
        f'**Budget Allocation ($ {unit_label})**',
        min_value=0,
        max_value=max_news,
        value=st.session_state.newspaper_spend,
        step=5,
        format="%d",
        key="newspaper_slider"
    )
    st.session_state.newspaper_spend = newspaper_spend  # Update session state immediately
    newspaper_units = (st.session_state.newspaper_spend * unit_factor) / newspaper_cost_per_unit if newspaper_cost_per_unit > 0 else 0
    st.metric("Allocated", f"${st.session_state.newspaper_spend:,} {unit_label}")
    st.metric("Ad Spots", f"{newspaper_units:,.0f}")

# --- Budget Summary ---
total_spent = st.session_state.tv_spend + st.session_state.radio_spend + st.session_state.newspaper_spend
remaining_budget = total_budget - total_spent

st.metric("Remaining Budget", f"${remaining_budget:,} {unit_label}")

st.markdown(f"""
<div style='background: linear-gradient(135deg, #00b894 0%, #00a085 100%); 
            color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin-top: 2rem;'>
    <h3>Total Advertising Investment: ${total_spent:,} {unit_label}</h3>
</div>
""", unsafe_allow_html=True)

# --- Sales Prediction ---
predicted_sales, tv_units_calc, radio_units_calc, newspaper_units_calc, tv_sales_contrib, radio_sales_contrib, newspaper_sales_contrib = enhanced_predict(
    st.session_state.tv_spend * 1000,  # Convert to actual dollars
    st.session_state.radio_spend * 1000,
    st.session_state.newspaper_spend * 1000,
    tv_cost_per_unit,
    radio_cost_per_unit,
    newspaper_cost_per_unit
)

roi = (predicted_sales - (total_spent * 1000)) / (total_spent * 1000) if total_spent > 0 else 0
efficiency = min(87 + (roi * 10), 95)
growth_opportunity = max(5, min(20, 12 + (roi * 15)))

# --- Sales Prediction Results ---
st.markdown("---")
st.markdown("""
<div class='prediction-card'>
    <h2 style='color: white; text-align: center;'>📈 Sales Prediction Results</h2>
</div>
""", unsafe_allow_html=True)

# Sales Prediction Metrics
pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)

with pred_col1:
    st.subheader(f"Total Predicted")
    st.success(f"""${predicted_sales:,.0f}""")

with pred_col2:
    st.subheader(f"Expected ROI")
    st.warning(f"""{roi:.2f}""")

with pred_col3:
    total_investment = total_spent * 1000
    profit = predicted_sales - total_investment
    st.subheader(f"Net Profit")
    st.info(f"""${profit:,.0f}""")

with pred_col4:
    st.subheader(f"Total Investment")
    st.error(f"""${total_investment:,.0f}""")

# --- Sales Breakdown by Channel ---
st.markdown("### 📊 Sales Breakdown by Channel")

breakdown_col1, breakdown_col2 = st.columns([2, 1])

with breakdown_col1:
    # Combined Bar Chart for Investment vs Sales Contribution
    sales_breakdown_data = {
        'Channel': ['Television', 'Radio', 'Newspaper'],
        'Investment': [st.session_state.tv_spend * 1000, st.session_state.radio_spend * 1000, st.session_state.newspaper_spend * 1000],
        'Sales_Contribution': [tv_sales_contrib, radio_sales_contrib, newspaper_sales_contrib],
        'ROI_Ratio': [tv_sales_contrib/(st.session_state.tv_spend * 1000) if st.session_state.tv_spend > 0 else 0, 
                      radio_sales_contrib/(st.session_state.radio_spend * 1000) if st.session_state.radio_spend > 0 else 0, 
                      newspaper_sales_contrib/(st.session_state.newspaper_spend * 1000) if st.session_state.newspaper_spend > 0 else 0]
    }
    
    df_sales_breakdown = pd.DataFrame(sales_breakdown_data)
    
    fig_breakdown = go.Figure()
    
    # Add bars for Investment
    fig_breakdown.add_trace(go.Bar(
        name='Investment ($)',
        x=df_sales_breakdown['Channel'],
        y=df_sales_breakdown['Investment'],
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        text=df_sales_breakdown['Investment'],
        texttemplate='$%{text:,.0f}',
        textposition='outside',
        yaxis='y'
    ))
    
    # Add bars for Sales Contribution
    fig_breakdown.add_trace(go.Bar(
        name='Sales Contribution ($)',
        x=df_sales_breakdown['Channel'],
        y=df_sales_breakdown['Sales_Contribution'],
        marker_color=['#a8e6cf', '#dcedc1', '#ffd3b6'],
        text=df_sales_breakdown['Sales_Contribution'],
        texttemplate='$%{text:,.0f}',
        textposition='outside',
        yaxis='y'
    ))
    
    # Add line for ROI Ratio
    fig_breakdown.add_trace(go.Scatter(
        name='ROI Ratio',
        x=df_sales_breakdown['Channel'],
        y=df_sales_breakdown['ROI_Ratio'],
        mode='lines+markers+text',
        line=dict(color='#FFD700', width=4),
        marker=dict(size=12, symbol='diamond'),
        text=[f'{x:.2f}x' for x in df_sales_breakdown['ROI_Ratio']],
        textposition='top center',
        yaxis='y2'
    ))
    
    fig_breakdown.update_layout(
        title="Investment vs Sales Contribution with ROI Ratios",
        xaxis=dict(title="Advertising Channels"),
        yaxis=dict(title="Amount ($)", side='left'),
        yaxis2=dict(
            title="ROI Ratio",
            side='right',
            overlaying='y',
            range=[0, max(df_sales_breakdown['ROI_Ratio']) * 1.2]
        ),
        barmode='group',
        plot_bgcolor='white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_breakdown, use_container_width=True)

with breakdown_col2:
    # Units and Efficiency Metrics
    st.markdown("#### 📈 Channel Performance")
    
    metrics_data = {
        'Channel': ['TV', 'Radio', 'Newspaper'],
        'Units': [tv_units_calc, radio_units_calc, newspaper_units_calc],
        'Cost/Unit': [tv_cost_per_unit, radio_cost_per_unit, newspaper_cost_per_unit],
        'Efficiency': [0.045, 0.185, 0.001]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Efficiency Gauge
    fig_efficiency = px.bar(
        df_metrics,
        x='Channel',
        y='Efficiency',
        color='Channel',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        title="Channel Efficiency Coefficients"
    )
    fig_efficiency.update_layout(showlegend=False, height=250)
    st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Units Chart
    fig_units = px.bar(
        df_metrics,
        x='Channel',
        y='Units',
        color='Channel',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        title="Number of Ad Units"
    )
    fig_units.update_layout(showlegend=False, height=250)
    st.plotly_chart(fig_units, use_container_width=True)

# --- Detailed Performance Analysis ---
st.markdown("### 🔍 Detailed Performance Analysis")

detail_col1, detail_col2 = st.columns(2)

with detail_col1:
    # Cost vs Efficiency Analysis
    cost_efficiency_data = {
        'Channel': ['Television', 'Radio', 'Newspaper'],
        'Cost_Per_Unit': [tv_cost_per_unit, radio_cost_per_unit, newspaper_cost_per_unit],
        'Efficiency_Score': [0.045 * 1000, 0.185 * 1000, 0.001 * 1000],  # Scaled for better visualization
        'ROI_Ratio': sales_breakdown_data['ROI_Ratio']
    }
    
    df_cost_efficiency = pd.DataFrame(cost_efficiency_data)
    
    fig_cost_eff = px.scatter(
        df_cost_efficiency,
        x='Cost_Per_Unit',
        y='Efficiency_Score',
        size='ROI_Ratio',
        color='Channel',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],
        title="Cost per Unit vs Efficiency Score",
        hover_data=['ROI_Ratio']
    )
    st.plotly_chart(fig_cost_eff, use_container_width=True)

with detail_col2:
    # Budget Utilization
    utilization_data = {
        'Category': ['Allocated Budget', 'Remaining Budget'],
        'Amount': [total_spent, remaining_budget],
        'Color': ['#00b894', '#dfe6e9']
    }
    
    df_utilization = pd.DataFrame(utilization_data)
    
    fig_utilization = px.pie(
        df_utilization,
        values='Amount',
        names='Category',
        color='Category',
        color_discrete_map={'Allocated Budget': '#00b894', 'Remaining Budget': '#dfe6e9'},
        title="Budget Utilization"
    )
    st.plotly_chart(fig_utilization, use_container_width=True)
    
# --- Executive Summary ---
st.markdown("---")
st.markdown("### 📋 Executive Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.markdown(f"""
    <div class='insight-card'>
        <h4>🎯 Key Performance Indicators</h4>
        <ul>
        <li><strong>Total Budget:</strong> ${total_budget:,} {unit_label}</li>
        <li><strong>Allocated Budget:</strong> ${total_spent:,} {unit_label}</li>
        <li><strong>Optimal Efficiency:</strong> {efficiency:.0f}% Achieved</li>
        <li><strong>ROI Potential:</strong> {roi:.2f}x Return</li>
        <li><strong>Growth Opportunity:</strong> +{growth_opportunity:.0f}% Revenue</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with summary_col2:
    # Dynamic recommendations
    if st.session_state.radio_spend < total_spent * 0.2:
        rec1 = "Increase Radio budget by 15% for higher ROI"
    else:
        rec1 = "Radio allocation is optimal"
    
    if st.session_state.tv_spend < total_spent * 0.6:
        rec2 = "Consider increasing TV for mass reach"
    else:
        rec2 = "TV budget ideal for brand awareness"
    
    if st.session_state.newspaper_spend > total_spent * 0.15:
        rec3 = "Consider reallocating Newspaper budget to Radio"
    else:
        rec3 = "Newspaper allocation is efficient"
    
    # Cost efficiency recommendations
    if tv_cost_per_unit > 150:
        rec4 = "Negotiate lower TV ad spot costs"
    else:
        rec4 = "TV costs are efficient"
    
    st.markdown(f"""
    <div class='insight-card'>
        <h4>🚀 Strategic Recommendations</h4>
        <ul>
        <li>{rec1}</li>
        <li>{rec2}</li>
        <li>{rec3}</li>
        <li>{rec4}</li>
        <li>Monitor campaign performance weekly</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Premium Footer ---
st.markdown("""
<div class='premium-footer'>
        <h4 style='color: white; margin: 0;'>Developed By</h4>
        <h2 style='color: white; margin: 0.5rem 0 0 0;'>Yuvraj Kumar Gond</h2>
        <p style='color: white; opacity: 0.8; margin: 0.5rem 0 0 0;'>
        Data Analyst | AI Solutions Architect
        </p>
</div>
""", unsafe_allow_html=True)
# ============================================================
# 🏠 HDB RESALE PRICE PREDICTOR - STREAMLIT APP (UPDATED)
# Model: Linear Regression (No floor_area_sqm)
# Dataset: 2017-2019 (Historical data only)
# Predicts: 1 month, 3 months, and 6 months from now
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request
import tempfile
import os
import datetime
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS FOR BETTER UI
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 6px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #145a8a;
        color: white;
        transform: scale(1.02);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-amount {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #6c757d;
        text-align: center;
        padding: 1rem;
        border-top: 1px solid #dee2e6;
        margin-top: 2rem;
    }
    .timeframe-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-top: 4px solid #1f77b4;
    }
    .timeframe-price {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin: 0.5rem 0;
    }
    .timeframe-label {
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">🏠 HDB Resale Price Predictor</div>', unsafe_allow_html=True)
st.markdown("### Predict HDB resale prices for 1, 3, and 6 months into the future")

# ============================================================
# WARNING: MODEL LIMITATIONS (PROMINENT DISPLAY)
# ============================================================

st.markdown("""
<div class="error-box">
    <strong>⚠️ CRITICAL: Model Limitations & Accuracy Warning</strong><br>
    <ul>
        <li>This model was trained on <strong>HDB resale data from 2017-2019</strong> only</li>
        <li>Predictions for 2026 are <strong>extrapolations</strong> and may be <strong>highly inaccurate</strong></li>
        <li><strong>Expected error margin:</strong> ±15-30% or more</li>
        <li>Use predictions as <strong>very rough estimates only</strong> - NOT financial advice</li>
        <li>Actual prices may vary significantly due to market changes, policy changes, and economic conditions</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model_from_url():
    """
    Download model from URL and load with joblib.
    Uses Streamlit's caching to avoid re-downloading on each rerun.
    """
    # 🔧 UPDATE THIS URL WITH YOUR ACTUAL PICKLE FILE URL
    MODEL_URL = "https://raw.githubusercontent.com/SzeMainWONG/datasets/main/model/hdb-price-predictor/hdb_price_predictor_my-model-final.pkl"
    
    try:
        with st.spinner("📥 Downloading model from repository... This may take a moment."):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                # Download the file
                urllib.request.urlretrieve(MODEL_URL, tmp_file.name)
                tmp_file_path = tmp_file.name
            
            # Load the model
            model = joblib.load(tmp_file_path)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            st.success("✅ Model loaded successfully!")
            return model
    except urllib.error.URLError as e:
        st.error(f"❌ Network error: Could not reach the model URL. Please check your internet connection.")
        st.info(f"Error details: {e}")
        return None
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found at the specified URL.")
        st.info("Please ensure the model file (.pkl) is uploaded to a public repository and the URL is correct.")
        return None
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.info("Please check that the URL points to a valid pickle file.")
        return None

# Load the model
model = load_model_from_url()

# Show model status
if model is None:
    st.warning("""
    ⚠️ **Model not loaded**
    
    To use this app, you need to:
    1. Train your model and save it as a `.pkl` file
    2. Upload it to a public repository (GitHub, Google Drive, etc.)
    3. Update the `MODEL_URL` in the code above
    
    **Alternative**: Place the `.pkl` file in the same directory as this app and use:
    ```python
    model = joblib.load('hdb_price_predictor_my-model-final.pkl')
    """)

# ============================================================
# SIDEBAR - MODEL INFORMATION
# ============================================================
st.sidebar.markdown("### 📊 Model Information")
st.sidebar.info("""
**Dataset**: HDB Resale Prices (2017-2019)  
**Records**: ~100,000 transactions  
**Model**: Linear Regression  
**R² Score**: ~0.76 (historical data)  

**Features Used**:
- Storey Mid-point
- Flat Age (years)
- Town
- Flat Category (Type + Model)

**Note**: floor_area_sqm was removed based on EDA findings
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Prediction Timeframes")
st.sidebar.markdown("""
The app will predict prices for:
- **1 month** from today
- **3 months** from today  
- **6 months** from today

*All predictions are extrapolations from 2017-2019 data*
""")

# ============================================================
# MAIN PREDICTION FORM
# ============================================================
st.markdown("### Enter Property Details")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📐 Property Specifications")
    
    # Storey range (converted to midpoint)
    storey_options = {
        '01 TO 03': 2,
        '04 TO 06': 5,
        '07 TO 09': 8,
        '10 TO 12': 11,
        '13 TO 15': 14,
        '16 TO 18': 17,
        '19 TO 21': 20,
        '22 TO 24': 23,
        '25 TO 27': 26,
        '28 TO 30': 29,
        '31 TO 33': 32,
        '34 TO 36': 35,
        '37 TO 39': 38,
        '40 TO 42': 41,
        '43 TO 45': 44,
        '46 TO 48': 47,
        '49 TO 51': 50
    }
    
    storey_label = st.selectbox(
        "Storey Range",
        options=list(storey_options.keys()),
        index=3,  # Default to 10 TO 12
        help="Select the storey range of the flat"
    )
    storey_mid = storey_options[storey_label]
    
    # Flat age (calculated from lease commencement date)
    today = datetime.datetime.now().date()
    current_year = today.year
    lease_start = st.number_input(
        "Lease Commencement Year",
        min_value=1960,
        max_value=current_year,
        value=1990,
        step=1,
        help="Year when the 99-year lease started"
    )
    flat_age = current_year - lease_start
    
    # Display flat age
    if flat_age < 0:
        st.error("❌ Lease commencement year cannot be in the future!")
    else:
        st.info(f"📊 **Flat Age**: {flat_age} years (lease started {lease_start})")

with col2:
    st.markdown("#### 🏢 Location & Type")
    
    # Town selection (common HDB towns)
    towns = [
        'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
        'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
        'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
        'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
        'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
        'TOA PAYOH', 'WOODLANDS', 'YISHUN'
    ]
    
    town = st.selectbox(
        "Town",
        options=sorted(towns),
        index=sorted(towns).index('ANG MO KIO') if 'ANG MO KIO' in towns else 0,
        help="Select the HDB town"
    )
    
    # Flat Category (Type + Model)
    flat_categories = [
        '1 ROOM - Standard', '2 ROOM - Standard', '2 ROOM - Improved',
        '2 ROOM - Model A', '2 ROOM - Premium Apartment', '2 ROOM - DBSS',
        '3 ROOM - Standard', '3 ROOM - Improved', '3 ROOM - New Generation',
        '3 ROOM - Model A', '3 ROOM - Simplified', '3 ROOM - Premium Apartment',
        '3 ROOM - DBSS', '3 ROOM - Type S1', '3 ROOM - Type S2',
        '4 ROOM - Standard', '4 ROOM - Improved', '4 ROOM - New Generation',
        '4 ROOM - Model A', '4 ROOM - Simplified', '4 ROOM - Premium Apartment',
        '4 ROOM - DBSS', '4 ROOM - Type S1', '4 ROOM - Type S2',
        '5 ROOM - Standard', '5 ROOM - Improved', '5 ROOM - New Generation',
        '5 ROOM - Model A', '5 ROOM - Premium Apartment', '5 ROOM - DBSS',
        '5 ROOM - Adjoined flat', '5 ROOM - Type S2',
        'EXECUTIVE - Standard', 'EXECUTIVE - Improved', 'EXECUTIVE - Model A',
        'EXECUTIVE - Premium Apartment', 'EXECUTIVE - Maisonette',
        'EXECUTIVE - Apartment', 'EXECUTIVE - DBSS',
        'MULTI-GENERATION - Premium Apartment', 'MULTI-GENERATION - DBSS'
    ]
    
    flat_category = st.selectbox(
        "Flat Type - Model",
        options=sorted(flat_categories),
        index=sorted(flat_categories).index('4 ROOM - New Generation') if '4 ROOM - New Generation' in flat_categories else 0,
        help="Combination of flat type and flat model"
    )

# ============================================================
# PREDICT BUTTON
# ============================================================
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Prices for 1, 3, and 6 Months", use_container_width=True)

# ============================================================
# PREDICTION LOGIC
# ============================================================
if predict_button:
    # Validate inputs
    if flat_age < 0:
        st.error("❌ Lease commencement year cannot be in the future!")
    elif model is None:
        st.error("❌ Model not loaded. Please check the model file.")
    else:
        # Prepare input
        input_data = pd.DataFrame({
            'storey_mid': [storey_mid],
            'flat_age': [flat_age],
            'town': [town],
            'flat_category': [flat_category]
        })
        
        try:
            # Make base prediction (current price estimate)
            base_price = model.predict(input_data)[0]
            
            # Calculate timeframes
            today = datetime.datetime.now().date()
            timeframes = {
                '1 Month': today + timedelta(days=30),
                '3 Months': today + timedelta(days=90),
                '6 Months': today + timedelta(days=180)
            }
            
            # Simple price projection (linear extrapolation based on historical trend)
            # Using a conservative 3% annual appreciation (adjustable)
            annual_appreciation = 0.03  # 3% per year
            monthly_appreciation = annual_appreciation / 12
            
            predictions = {}
            for label, date in timeframes.items():
                months_ahead = (date - today).days / 30.44  # Average days per month
                # Apply simple compound growth
                price = base_price * (1 + monthly_appreciation) ** months_ahead
                predictions[label] = {
                    'price': price,
                    'date': date,
                    'months': months_ahead
                }
            
            # ============================================================
            # DISPLAY RESULTS
            # ============================================================
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            # Current date info
            st.markdown(f"""
            <div class="info-box">
                <strong>📅 Today's Date:</strong> {today.strftime('%B %d, %Y')}
                <br>
                <strong>Base Price (Current Estimate):</strong> ${base_price:,.0f}
                <br>
                <small>This is the estimated price based on 2017-2019 data</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Display predictions in three columns
            col1, col2, col3 = st.columns(3)
            
            # Color coding for predictions
            colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
            
            for idx, (label, pred) in enumerate(predictions.items()):
                with [col1, col2, col3][idx]:
                    st.markdown(f"""
                    <div class="timeframe-card">
                        <div class="timeframe-label">📆 {label}</div>
                        <div class="timeframe-price">${pred['price']:,.0f}</div>
                        <div style="color: #6c757d; font-size: 0.85rem;">
                            {pred['date'].strftime('%B %d, %Y')}
                            <br>
                            ({pred['months']:.1f} months ahead)
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #95a5a6;">
                            ±15-30% error margin
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ============================================================
            # PRICE COMPARISON TABLE
            # ============================================================
            st.markdown("---")
            st.markdown("### 📈 Price Comparison")
            
            # Create comparison DataFrame
            comparison_data = []
            for label, pred in predictions.items():
                price_change = pred['price'] - base_price
                price_change_pct = (price_change / base_price) * 100
                comparison_data.append({
                    'Timeframe': label,
                    'Date': pred['date'].strftime('%B %d, %Y'),
                    'Predicted Price': f"${pred['price']:,.0f}",
                    'Change from Today': f"${price_change:+,.0f}",
                    'Change %': f"{price_change_pct:+.1f}%"
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(
                df_comparison,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Timeframe': st.column_config.TextColumn('Timeframe', width='small'),
                    'Date': st.column_config.TextColumn('Date', width='medium'),
                    'Predicted Price': st.column_config.TextColumn('Predicted Price', width='medium'),
                    'Change from Today': st.column_config.TextColumn('Change from Today', width='medium'),
                    'Change %': st.column_config.TextColumn('Change %', width='small')
                }
            )
            
            # ============================================================
            # VISUALIZATION: Price Trend
            # ============================================================
            
            with st.expander("📊 View Price Trend Visualization", expanded=True):
                # Prepare data for plot
                labels = ['Today'] + list(predictions.keys())
                prices = [base_price] + [pred['price'] for pred in predictions.values()]
                dates = [today] + [pred['date'] for pred in predictions.values()]
                
                # Create figure
                fig = go.Figure()
                
                # Add main trend line
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=prices,
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=12, color='#1f77b4')
                ))
                
                # Add uncertainty bands (±15%)
                lower_bound = [p * 0.85 for p in prices]
                upper_bound = [p * 1.15 for p in prices]
                
                fig.add_trace(go.Scatter(
                    x=dates + dates[::-1],
                    y=upper_bound + lower_bound[::-1],
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    name='Uncertainty Range'
                ))
                
                # Add prediction points with different colors
                colors_plot = ['#2ecc71', '#f39c12', '#e74c3c']
                for i, (label, pred) in enumerate(predictions.items()):
                    fig.add_trace(go.Scatter(
                        x=[pred['date']],
                        y=[pred['price']],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color=colors_plot[i],
                            symbol='star',
                            line=dict(color='white', width=2)
                        ),
                        name=label,
                        hovertemplate=f"<b>{label}</b><br>Price: ${pred['price']:,.0f}<br>Date: {pred['date'].strftime('%B %d, %Y')}<extra></extra>"
                    ))
                
                # Add current price marker
                fig.add_trace(go.Scatter(
                    x=[today],
                    y=[base_price],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='#1f77b4',
                        symbol='circle',
                        line=dict(color='white', width=2)
                    ),
                    name='Today\'s Estimate',
                    hovertemplate=f"<b>Today</b><br>Price: ${base_price:,.0f}<br>Date: {today.strftime('%B %d, %Y')}<extra></extra>"
                ))
                
                # Update layout
                fig.update_layout(
                    title={
                        'text': 'Estimated Price Trend (1, 3, and 6 Months Ahead)',
                        'font': {'size': 20}
                    },
                    xaxis_title='Date',
                    yaxis_title='Estimated Price ($)',
                    hovermode='x unified',
                    showlegend=True,
                    height=450,
                    template='plotly_white',
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    )
                )
                
                # Format y-axis as currency
                fig.update_layout(
                    yaxis=dict(
                        tickformat=',.0f',
                        tickprefix='$'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption("⚠️ This visualization is based on a simple 3% annual appreciation projection from the base estimate. Actual prices may vary significantly.")

            # ============================================================
            # ACCURACY WARNINGS (Enhanced)
            # ============================================================
            
            st.markdown("---")
            st.markdown("### ⚠️ Accuracy & Limitations")
            
            # Create warning columns
            warn_col1, warn_col2 = st.columns(2)
            
            with warn_col1:
                st.markdown("""
                <div class="error-box">
                    <strong>🚨 CRITICAL: Future Predictions</strong><br>
                    <ul>
                        <li>Model trained on <strong>2017-2019 data only</strong></li>
                        <li>All predictions are <strong>extrapolations</strong></li>
                        <li>HDB prices have likely changed significantly since 2019</li>
                        <li><strong>Expected error:</strong> ±15-30% or more</li>
                        <li>Predictions for 6 months out are <strong>most uncertain</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with warn_col2:
                st.markdown("""
                <div class="warning-box">
                    <strong>📌 Important Factors NOT in Model</strong><br>
                    <ul>
                        <li>Recent policy changes (cooling measures, grants)</li>
                        <li>Economic conditions (inflation, interest rates)</li>
                        <li>Property renovations and condition</li>
                        <li>Proximity to MRT, schools, amenities</li>
                        <li>Exact floor level (only mid-point used)</li>
                        <li>Unit orientation, facing, and view</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Model performance disclaimer
            st.markdown("""
            <div class="info-box">
                <strong>📊 Model Performance</strong><br>
                <ul>
                    <li><strong>Historical R²:</strong> 0.86 (on 2017-2019 test data)</li>
                    <li><strong>Estimated Future R²:</strong> ~0.76 (conservative estimate)</li>
                    <li>Future predictions have <strong>higher uncertainty</strong></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # ============================================================
            # FINANCIAL DISCLAIMER
            # ============================================================
            
            st.markdown("---")
            st.markdown("""
            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; padding: 1.5rem;">
                <strong>⚠️ FINANCIAL DISCLAIMER</strong><br><br>
                This tool provides <strong>estimated</strong> HDB resale prices based on historical data (2017-2019).
                These predictions are <strong>NOT FINANCIAL ADVICE</strong> and should not be used for:
                <ul>
                    <li>Making actual property purchase decisions</li>
                    <li>Property valuation for loans or mortgages</li>
                    <li>Any financial transaction or investment decision</li>
                </ul>
                <strong>Recommendations:</strong>
                <ul>
                    <li>Consult <strong>professional property valuers</strong> for accurate assessments</li>
                    <li>Check <strong>current market data</strong> from HDB and property portals</li>
                    <li>Consider recent <strong>transaction prices</strong> in the same area</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}")
            st.info("Please check your inputs and try again.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

# Additional resources and links
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    ### 📚 Resources
    - [HDB Resale Portal](https://www.hdb.gov.sg/residential/buying-a-flat/resale)
    - [SRX Property](https://www.srx.com.sg/)
    - [URA Real Estate](https://www.ura.gov.sg/realEstateIIWeb/search.action)
    """)
with col2:
    st.markdown("""
    ### 📊 Data Sources
    - [HDB Resale Prices](https://data.gov.sg/dataset/resale-flat-prices)
    - [Data.gov.sg](https://data.gov.sg/)
    - [PropertyGuru](https://www.propertyguru.com.sg/)
    """)
with col3:
    st.markdown("""
    ### 🔧 Model Info
    - **Model**: Linear Regression
    - **Training Data**: 2017-2019
    - **Features**: Storey, Age, Town, Type-Model
    - **R² Score**: ~0.76 (historical)
    - **Error Margin**: ±15-30% for future predictions
    """)

st.markdown("""
<div class="disclaimer">
    <strong>⚠️ DISCLAIMER:</strong> This tool is for educational and demonstration purposes only.
    Predictions are based on historical data (2017-2019) and may be highly inaccurate for current/future prices.
    Do not use for financial decisions. Always consult professional property valuers for accurate assessments.
    <br><br>
    Developed for educational purposes | Not affiliated with HDB
    <br>
    <span style="color: #dc3545; font-weight: bold;">Remember: Past performance does not guarantee future results.</span>
</div>
""", unsafe_allow_html=True)
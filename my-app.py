#Model predicts for future six months based on timestamp.now() when button activated on the streamlit browser online
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="HDB Price Predictor (Today-Based Extrapolation)",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #DC2626;
        text-align: center;
        margin-bottom: 1rem;
        background-color: #FEF2F2;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #DC2626;
    }
    .prediction-box {
        background-color: #F0F4FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        text-align: center;
    }
    .warning-box {
        background-color: #FEF2F2;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #FCA5A5;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2563EB;
        color: white;
    }
    .timestamp-box {
        background-color: #F0FDF4;
        padding: 0.8rem;
        border-radius: 8px;
        border: 1px solid #86EFAC;
        text-align: center;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():
    try:
        with open('house_model_extrapolate.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please run 'model.py' first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_model()

# ============================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================

def extract_lease_years(lease_str):
    if pd.isna(lease_str):
        return np.nan
    parts = lease_str.split()
    years = 0
    months = 0
    for i, part in enumerate(parts):
        if part == 'years' or part == 'year':
            years = float(parts[i-1])
        elif part == 'months' or part == 'month':
            months = float(parts[i-1])
    return years + months / 12

def prepare_input_data(df):
    """Apply all feature engineering to input dataframe"""
    if 'storey_range' in df.columns:
        df['floor_level'] = df['storey_range'].str.split(" ").str[0].astype(float)
    if 'remaining_lease' in df.columns:
        df['remaining_lease_years'] = df['remaining_lease'].apply(extract_lease_years)
    if 'lease_commence_date' in df.columns:
        df['flat_age'] = 2026 - df['lease_commence_date']
    if 'month' in df.columns:
        df['month_dt'] = pd.to_datetime(df['month'])
        df['month_num'] = df['month_dt'].dt.year + df['month_dt'].dt.month / 12
    
    required_features = [
        'floor_area_sqm', 'floor_level', 'remaining_lease_years', 
        'flat_age', 'month_num', 'town', 'flat_type', 'flat_model'
    ]
    
    for feat in required_features:
        if feat not in df.columns:
            df[feat] = np.nan
    
    return df[required_features]

def generate_future_dates_from_timestamp(n_months=6):
    """
    Generate future month dates starting from next month
    based on the current timestamp at call time.
    """
    current_timestamp = pd.Timestamp.now().normalize()
    current_date_str = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    # Start from the first day of next month
    if current_timestamp.month == 12:
        start_date = pd.Timestamp(year=current_timestamp.year + 1, month=1, day=1)
    else:
        start_date = pd.Timestamp(year=current_timestamp.year, month=current_timestamp.month + 1, day=1)
    
    future_dates = []
    current = start_date
    for i in range(n_months):
        future_dates.append(current)
        # Add 1 month
        year = current.year + (current.month + 1) // 13
        month = (current.month + 1) % 13
        if month == 0:
            month = 12
        current = pd.Timestamp(year=year, month=month, day=1)
    
    return future_dates, current_timestamp, current_date_str

# ============================================
# MAIN HEADER
# ============================================

st.markdown('<p class="main-header">🔮 HDB Resale Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">⚠️ EDUCATIONAL PROJECT - Today-Based Extrapolation (2019 → Present)</p>', unsafe_allow_html=True)

# Show warning
with st.expander("⚠️ Important: Understanding Extrapolation Limitations", expanded=True):
    st.markdown("""
    **This is a LEARNING PROJECT only. Predictions are HIGHLY SPECULATIVE!**
    
    The model was trained on **2017-2019 HDB data** and is extrapolating **5+ years into the future**.
    
    **Why predictions may be inaccurate:**
    - ❌ COVID-19 pandemic (2020-2022) is NOT captured
    - ❌ Government cooling measures (2018, 2021, 2022, 2023) are NOT captured
    - ❌ Interest rate changes and inflation are NOT captured
    - ❌ Supply and demand shifts are NOT captured
    - ❌ HDB policy changes are NOT captured
    
    **Use this for:**
    - ✅ Learning how extrapolation works
    - ✅ Understanding machine learning limitations
    - ✅ Experimenting with time series forecasting
    - ✅ Educational demonstrations
    
    **DO NOT use for:**
    - ❌ Real estate investment decisions
    - ❌ Financial planning
    - ❌ Property valuation
    """)

# ============================================
# SIDEBAR - INPUT
# ============================================

st.sidebar.markdown("## 🏗️ Property Details")
st.sidebar.markdown("---")

with st.sidebar.form("prediction_form"):
    town = st.selectbox(
        "📍 Town",
        ["ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH", 
         "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG", 
         "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST", 
         "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", 
         "SENGKANG", "SERANGOON", "TAMPINES", "TOA PAYOH", "WOODLANDS", 
         "YISHUN"]
    )
    
    flat_type = st.selectbox(
        "🏠 Flat Type",
        ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "1 ROOM", "MULTI-GENERATION"]
    )
    
    flat_model = st.selectbox(
        "🏢 Flat Model",
        ["New Generation", "Improved", "Standard", "Model A", "Simplified", 
         "Apartment", "Maisonette", "Premium Apartment", "DBSS", "Model A2", 
         "Adjoined flat", "Type S1", "Type S2"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        floor_area_sqm = st.number_input(
            "📐 Floor Area (sqm)",
            min_value=20.0,
            max_value=300.0,
            value=90.0,
            step=1.0
        )
    
    with col2:
        storey_range = st.selectbox(
            "🏗️ Storey Range",
            ["01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12", "13 TO 15", 
             "16 TO 18", "19 TO 21", "22 TO 24", "25 TO 27", "28 TO 30", 
             "31 TO 33", "34 TO 36", "37 TO 39", "40 TO 42", "43 TO 45", 
             "46 TO 48", "49 TO 51"]
        )
    
    col3, col4 = st.columns(2)
    with col3:
        lease_commence_date = st.number_input(
            "📅 Lease Commence Year",
            min_value=1960,
            max_value=2026,
            value=1980,
            step=1
        )
    
    with col4:
        remaining_lease = st.text_input(
            "⏳ Remaining Lease",
            value="61 years 06 months",
            help="Format: 'X years Y months'"
        )
    
    # Show note about when prediction will be based
    st.sidebar.markdown("---")
    st.sidebar.markdown("📆 **Prediction will use:**")
    st.sidebar.markdown("*Current timestamp when you click 'Predict'*")
    
    submitted = st.form_submit_button("🔮 Predict Price (Extrapolated)")

# ============================================
# PREDICTION
# ============================================

if model is None:
    st.warning("⚠️ Model not loaded. Please run `model.py` first.")
    st.stop()

if submitted:
    with st.spinner("🧠 Extrapolating into the future..."):
        try:
            # ============================================
            # GET CURRENT TIMESTAMP AT PREDICTION TIME
            # ============================================
            future_dates, current_timestamp, current_date_str = generate_future_dates_from_timestamp(6)
            
            # Show when the prediction was made
            st.markdown(f"""
            <div class="timestamp-box">
                ⏰ Prediction generated on: <b>{current_date_str}</b><br>
                📆 Predicting for: <b>{future_dates[0].strftime('%Y-%m')}</b> to <b>{future_dates[-1].strftime('%Y-%m')}</b>
            </div>
            """, unsafe_allow_html=True)
            
            predictions = []
            
            for month_dt in future_dates:
                month_str = month_dt.strftime('%Y-%m')
                
                # Create input data
                input_data = pd.DataFrame({
                    'month': [month_str],
                    'town': [town],
                    'flat_type': [flat_type],
                    'flat_model': [flat_model],
                    'floor_area_sqm': [floor_area_sqm],
                    'storey_range': [storey_range],
                    'lease_commence_date': [lease_commence_date],
                    'remaining_lease': [remaining_lease]
                })
                
                # Apply feature engineering
                input_processed = prepare_input_data(input_data)
                
                # Predict
                pred = model.predict(input_processed)[0]
                predictions.append({
                    'month': month_dt,
                    'month_str': month_str,
                    'price': pred
                })
            
            # ============================================
            # DISPLAY RESULTS
            # ============================================
            
            # Show extrapolation warning
            st.markdown("""
            <div class="warning-box">
                ⚠️ <b>EXTRAPOLATION ALERT</b>: These predictions are based on 2017-2019 trends extrapolated 
                to present day. Real-world prices may differ significantly due to major events not captured in the training data.
            </div>
            """, unsafe_allow_html=True)
            
            # Show price trend as metrics
            st.markdown("## 📈 Predicted Price Trend (6 Months)")
            
            # Display as metrics
            cols = st.columns(6)
            for i, row in enumerate(predictions):
                with cols[i]:
                    st.metric(
                        label=row['month_str'],
                        value=f"${row['price']:,.0f}"
                    )
            
            # Show detailed table
            st.markdown("## 📊 Detailed Forecast")
            display_df = pd.DataFrame(predictions)[['month_str', 'price']].copy()
            display_df.columns = ['Month', 'Predicted Price']
            display_df['Predicted Price'] = display_df['Predicted Price'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_df, hide_index=True, use_container_width=True)
            
            # ============================================
            # SHOW PLOTLY CHART (Fixed for older versions)
            # ============================================
            st.markdown("## 📊 Price Trend Visualization")
            
            # Create figure
            fig = go.Figure()
            
            # Extract data
            months = [p['month'] for p in predictions]
            prices = [p['price'] for p in predictions]
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=months,
                y=prices,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='#1E3A8A', width=3),
                marker=dict(size=10, color='#1E3A8A')
            ))
            
            # Add confidence interval (8% uncertainty)
            uncertainty_pct = 0.08
            upper = [p * (1 + uncertainty_pct) for p in prices]
            lower = [p * (1 - uncertainty_pct) for p in prices]
            
            # Upper bound
            fig.add_trace(go.Scatter(
                x=months,
                y=upper,
                mode='lines',
                name='Upper Bound (+8%)',
                line=dict(color='rgba(30, 58, 138, 0.3)', width=1, dash='dash')
            ))
            
            # Lower bound with fill
            fig.add_trace(go.Scatter(
                x=months,
                y=lower,
                mode='lines',
                name='Lower Bound (-8%)',
                line=dict(color='rgba(30, 58, 138, 0.3)', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(30, 58, 138, 0.1)'
            ))
            
            # Update layout (using the older method for compatibility)
            fig.update_layout(
                title=f"{town} - {flat_type} ({floor_area_sqm} sqm)",
                xaxis_title="Month",
                yaxis_title="Predicted Price (SGD)",
                hovermode='x unified',
                template='plotly_white',
                height=500,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                # Format y-axis as currency - OLDER METHOD
                yaxis=dict(
                    tickprefix="$",
                    tickformat=",.0f"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ============================================
            # SHOW INDIVIDUAL PROPERTY DETAILS
            # ============================================
            
            st.markdown("## 📋 Property Details")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("🏠 Town", town)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("📐 Floor Area", f"{floor_area_sqm} sqm")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("🏢 Flat Type", flat_type)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("📅 Lease Year", lease_commence_date)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # ============================================
            # CALCULATE AND SHOW ADDITIONAL INSIGHTS
            # ============================================
            
            st.markdown("## 💡 Key Insights")
            
            # Calculate price change
            first_price = predictions[0]['price']
            last_price = predictions[-1]['price']
            price_change = last_price - first_price
            pct_change = (price_change / first_price) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "📈 Price Change (6 months)",
                    f"${price_change:,.2f}",
                    delta=f"{pct_change:+.2f}%"
                )
            
            with col2:
                # Calculate average monthly growth
                monthly_growth = price_change / 6
                st.metric(
                    "📊 Average Monthly Change",
                    f"${monthly_growth:,.2f}",
                    delta=f"{((monthly_growth / first_price) * 100):+.2f}%"
                )
            
            with col3:
                # Price per sqm
                price_per_sqm = first_price / floor_area_sqm
                st.metric(
                    "💰 Price per sqm",
                    f"${price_per_sqm:,.2f}"
                )
            
            # ============================================
            # DISCLAIMER
            # ============================================
            
            st.markdown("""
            <div style="text-align: center; color: #6B7280; font-size: 0.8rem; margin-top: 2rem; padding: 1rem; background-color: #F3F4F6; border-radius: 8px;">
                <p><b>⚠️ EDUCATIONAL USE ONLY</b></p>
                <p>This is a learning project demonstrating time series extrapolation. 
                Predictions are based on 2017-2019 data and do NOT account for:</p>
                <p style="font-size: 0.75rem;">
                    • COVID-19 impacts • Government cooling measures • Interest rate changes<br>
                    • Inflation • Policy changes • Market supply/demand shifts
                </p>
                <p style="font-weight: bold; color: #DC2626;">DO NOT use for real investment decisions.</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that all input fields are filled correctly.")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
    <p>🔮 HDB Resale Price Predictor (Today-Based Extrapolation) | Built with Streamlit & scikit-learn</p>
    <p>📚 Learning Project - Demonstrating Time Series Extrapolation</p>
    <p>⏰ Predictions use the current timestamp when you click "Predict"</p>
</div>
""", unsafe_allow_html=True)
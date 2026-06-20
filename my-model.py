#Model predicts for future six months based on timestamp.now() when button activated on the streamlit browser online
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD DATA
# ============================================

#url = "https://raw.githubusercontent.com/kohjiaxuan/Predicting-HDB-Price-with-Machine-Learning/master/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv"
#url = "https://raw.githubusercontent.com/SzeMainWONG/datasets/main/dataset/hdb_resale_prices.csv"
url = "https://media.githubusercontent.com/media/SzeMainWONG/datasets/main/dataset/hdb_resale_prices.csv"
data = pd.read_csv(url)

print(f"📥 Loaded {len(data):,} records")
print(f"📅 Date range: {data['month'].min()} to {data['month'].max()}")

# Convert month to datetime
data['month_dt'] = pd.to_datetime(data['month'])
data['month_num'] = data['month_dt'].dt.year + data['month_dt'].dt.month / 12

# ============================================
# 2. FEATURE ENGINEERING
# ============================================

# Extract floor level from storey_range
data['floor_level'] = data['storey_range'].str.split(" ").str[0].astype(float)

# Extract remaining lease years
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

data['remaining_lease_years'] = data['remaining_lease'].apply(extract_lease_years)
data['flat_age'] = 2026 - data['lease_commence_date']

print(f"🔧 Engineered {len(data.columns)} total features")

# ============================================
# 3. CHRONOLOGICAL SPLIT
# ============================================

# Sort by date
data = data.sort_values('month_dt')

# Get unique months
unique_months = data['month_dt'].unique()
print(f"\n📅 Unique months in dataset: {len(unique_months)} months")
print(f"   First: {unique_months[0].strftime('%Y-%m')}")
print(f"   Last: {unique_months[-1].strftime('%Y-%m')}")

# Split: 80% training, 20% testing
split_idx = int(len(unique_months) * 0.8)
train_months = unique_months[:split_idx]
test_months = unique_months[split_idx:]

print(f"\n📊 Training months: {train_months[0].strftime('%Y-%m')} to {train_months[-1].strftime('%Y-%m')}")
print(f"📊 Test months: {test_months[0].strftime('%Y-%m')} to {test_months[-1].strftime('%Y-%m')}")

# Split data
train_data = data[data['month_dt'].isin(train_months)]
test_data = data[data['month_dt'].isin(test_months)]

print(f"\n📊 Training data: {len(train_data):,} records")
print(f"📊 Test data: {len(test_data):,} records")

# ============================================
# 4. FEATURE SELECTION
# ============================================

numeric_features = [
    'floor_area_sqm',
    'floor_level',
    'remaining_lease_years',
    'flat_age',
    'month_num'  # Key feature for time trend
]

categorical_features = [
    'town',
    'flat_type',
    'flat_model'
]

target = 'resale_price'

print(f"\n📌 Numeric features: {numeric_features}")
print(f"📌 Categorical features: {categorical_features}")

# ============================================
# 5. CREATE PREPROCESSOR
# ============================================

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
])

# ============================================
# 6. BUILD PIPELINE
# ============================================

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=0.1, random_state=42))
])

print("\n🔧 Model: Ridge Regression with L2 regularization")

# ============================================
# 7. TRAIN MODEL
# ============================================

X_train = train_data.drop(target, axis=1)
y_train = train_data[target]

print("\n🏋️ Training model...")
model.fit(X_train, y_train)

# ============================================
# 8. EVALUATE WITH MULTIPLE METRICS
# ============================================

X_test = test_data.drop(target, axis=1)
y_test = test_data[target]

if len(X_test) > 0:
    y_pred = model.predict(X_test)
    
    # Calculate all metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    pct_error = (mae / y_test.mean()) * 100
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
    
    print(f"\n📊 Model Performance on Test Data:")
    print("="*50)
    print(f"  Mean Absolute Error (MAE):       ${mae:,.2f}")
    print(f"  Mean Squared Error (MSE):        ${mse:,.2f}")
    print(f"  Root Mean Squared Error (RMSE):  ${rmse:,.2f}")
    print(f"  R² Score:                         {r2:.4f}")
    print(f"  Average Percentage Error (MAE%):  {pct_error:.2f}%")
    print(f"  Mean Absolute Percentage Error:   {mape:.2f}%")
    print("="*50)
    
    # Explain what RMSE means
    print(f"\n📖 RMSE Interpretation:")
    print(f"  • On average, predictions are off by ${rmse:,.2f}")
    print(f"  • RMSE penalizes large errors more than MAE (squaring amplifies big mistakes)")
    print(f"  • RMSE is {rmse/mae:.2f}x larger than MAE, indicating some significant outliers")
    
    # Calculate error distribution
    errors = y_pred - y_test.values
    abs_errors = np.abs(errors)
    
    print(f"\n📊 Error Distribution:")
    print(f"  • Min Error:   ${errors.min():,.2f}")
    print(f"  • Max Error:   ${errors.max():,.2f}")
    print(f"  • Median Error: ${np.median(abs_errors):,.2f}")
    print(f"  • 75th Percentile: ${np.percentile(abs_errors, 75):,.2f}")
    print(f"  • 90th Percentile: ${np.percentile(abs_errors, 90):,.2f}")
    
    # Sample predictions vs actuals - FIXED VERSION
    print(f"\n🔍 Sample Predictions vs Actuals (10 random test samples):")
    n_samples = min(10, len(y_test))
    sample_indices = np.random.choice(len(y_test), n_samples, replace=False)
    
    # Convert to numpy arrays for easier indexing
    y_test_array = y_test.values
    y_pred_array = y_pred
    
    sample_df = pd.DataFrame({
        'Actual': y_test_array[sample_indices],
        'Predicted': y_pred_array[sample_indices],
        'Error': errors[sample_indices],
        'Abs_Error': abs_errors[sample_indices]
    })
    sample_df = sample_df.round(2)
    print(sample_df.to_string(index=False))
    
    # For the sample, show which town/flat type
    print(f"\n🔍 Sample Predictions with Property Details:")
    sample_full = test_data.iloc[sample_indices][['town', 'flat_type', 'floor_area_sqm', 'resale_price']].copy()
    sample_full['Predicted'] = y_pred_array[sample_indices]
    sample_full['Error'] = errors[sample_indices]
    sample_full['Abs_Error'] = abs_errors[sample_indices]
    sample_full = sample_full.round(2)
    print(sample_full.to_string(index=False))

# ============================================
# 9. SAVE MODEL
# ============================================

with open('house_model_extrapolate.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n💾 Model saved as 'house_model_extrapolate.pkl'")

# ============================================
# 10. GENERATE 6-MONTH FORECAST FROM TODAY
# ============================================

print("\n🔮 Generating 6-month forecast from TODAY...")

def generate_future_dates_from_today(n_months=6):
    """Generate future month dates starting from next month"""
    today = pd.Timestamp.now().normalize()
    if today.month == 12:
        start_date = pd.Timestamp(year=today.year + 1, month=1, day=1)
    else:
        start_date = pd.Timestamp(year=today.year, month=today.month + 1, day=1)
    
    future_dates = []
    current = start_date
    for i in range(n_months):
        future_dates.append(current)
        year = current.year + (current.month + 1) // 13
        month = (current.month + 1) % 13
        if month == 0:
            month = 12
        current = pd.Timestamp(year=year, month=month, day=1)
    return future_dates

# Generate future dates from today
future_dates = generate_future_dates_from_today(n_months=6)
today = pd.Timestamp.now().normalize()

print(f"📅 Today's date: {today.strftime('%Y-%m-%d')}")
print(f"📅 Future dates: {[d.strftime('%Y-%m') for d in future_dates]}")

# Show warning about extrapolation
print("\n⚠️  EXTRAPOLATION WARNING:")
print("  • Model was trained on 2017-2019 data")
print("  • Predicting beyond this range assumes trends continue unchanged")
print("  • Use these predictions for LEARNING/EDUCATIONAL purposes ONLY")
print("  • Real-world prices may differ significantly due to:")
print("    - COVID-19 pandemic impacts")
print("    - Government cooling measures (2018, 2021, 2022, 2023)")
print("    - Economic changes (inflation, interest rates)")
print("    - Supply and demand shifts")

# Get all unique towns
towns = train_data['town'].unique()
print(f"\n🏘️ Number of towns: {len(towns)}")

# Create future data for each town
future_predictions = []

for town in towns[:10]:  # Limit to first 10 towns for demo
    # Create a sample flat (4-room, New Generation, 90 sqm, 10-12 floor)
    future_data = pd.DataFrame({
        'month': [d.strftime('%Y-%m') for d in future_dates],
        'month_dt': future_dates,
        'town': [town] * 6,
        'flat_type': ['4 ROOM'] * 6,
        'flat_model': ['New Generation'] * 6,
        'floor_area_sqm': [90] * 6,
        'storey_range': ['10 TO 12'] * 6,
        'lease_commence_date': [1980] * 6,
        'remaining_lease': ['61 years 06 months'] * 6,
        'block': [''] * 6,
        'street_name': [''] * 6
    })
    
    # Apply feature engineering
    future_data['floor_level'] = future_data['storey_range'].str.split(" ").str[0].astype(float)
    future_data['remaining_lease_years'] = future_data['remaining_lease'].apply(extract_lease_years)
    future_data['flat_age'] = 2026 - future_data['lease_commence_date']
    future_data['month_num'] = future_data['month_dt'].dt.year + future_data['month_dt'].dt.month / 12
    
    # Predict
    try:
        preds = model.predict(future_data)
        for i, pred in enumerate(preds):
            future_predictions.append({
                'town': town,
                'flat_type': '4 ROOM',
                'flat_model': 'New Generation',
                'floor_area_sqm': 90,
                'month': future_dates[i],
                'predicted_price': pred
            })
    except Exception as e:
        print(f"  ⚠️ Error predicting for {town}: {str(e)}")

# Convert to DataFrame
future_df = pd.DataFrame(future_predictions)

if len(future_df) > 0:
    # Pivot for easier viewing
    future_pivot = future_df.pivot(index='month', columns='town', values='predicted_price')
    
    print(f"\n📈 6-Month Forecast from TODAY (4-Room, 90 sqm, New Generation):")
    print("="*80)
    print(future_pivot.round(2))
    
    # Save forecast with today's date in filename
    filename = f'6_month_forecast_from_{pd.Timestamp.now().strftime("%Y-%m-%d")}.csv'
    future_pivot.to_csv(filename)
    print(f"\n💾 Forecast saved to '{filename}'")
else:
    print("\n⚠️ No future predictions generated")

# ============================================
# 11. COMPARE WITH HISTORICAL TREND
# ============================================

print("\n📊 Historical vs Predicted (ANG MO KIO Example):")
print("-"*50)

# Get historical prices for ANG MO KIO (4-room, New Generation)
amk_historical = train_data[
    (train_data['town'] == 'ANG MO KIO') & 
    (train_data['flat_type'] == '4 ROOM') & 
    (train_data['flat_model'] == 'New Generation')
].copy()

# Group by month
amk_monthly = amk_historical.groupby('month_dt')['resale_price'].mean().reset_index()

if len(amk_monthly) > 0:
    print(f"Historical prices (last 6 months in training):")
    for _, row in amk_monthly.tail(6).iterrows():
        print(f"  {row['month_dt'].strftime('%Y-%m')}: ${row['resale_price']:,.2f}")
    
    # Get future predictions for ANG MO KIO
    amk_future = future_df[future_df['town'] == 'ANG MO KIO']
    if len(amk_future) > 0:
        print(f"\nPredicted prices (next 6 months from today):")
        for _, row in amk_future.iterrows():
            print(f"  {row['month'].strftime('%Y-%m')}: ${row['predicted_price']:,.2f}")

# ============================================
# 12. KEY INSIGHTS
# ============================================

print("\n💡 Key Insights:")
print("-"*50)
print(f"✅ Model trained on {len(train_data):,} records (2017-2019)")
print(f"✅ Tested on {len(test_data):,} held-out records")

if len(X_test) > 0:
    print(f"\n📊 Performance Metrics Summary:")
    print(f"  • MAE:  ${mae:,.2f} (average error in dollars)")
    print(f"  • RMSE: ${rmse:,.2f} (penalizes large errors)")
    print(f"  • R²:   {r2:.4f} (variance explained)")
    print(f"  • MAPE: {mape:.2f}% (average percentage error)")
    
    print(f"\n📈 RMSE Interpretation:")
    if rmse > mae * 1.5:
        print("  ⚠️  RMSE is significantly larger than MAE - there are some large outliers")
    else:
        print("  ✅ RMSE is close to MAE - errors are relatively consistent")
    
    if r2 > 0.7:
        print(f"  ✅ R² of {r2:.4f} indicates the model explains most of the variance")
    else:
        print(f"  ⚠️  R² of {r2:.4f} indicates room for improvement")

print(f"\n📈 Prediction Details:")
print(f"  • Forecasting from: {future_dates[0].strftime('%Y-%m')} (next month)")
print(f"  • Forecasting to: {future_dates[-1].strftime('%Y-%m')} (6 months ahead)")

print("\n⚠️  CRITICAL LIMITATIONS FOR TODAY-BASED PREDICTION:")
print("  • Model is trained on 2017-2019 market conditions")
print("  • Extrapolating 5+ years into the future is HIGHLY UNRELIABLE")
print("  • Real-world factors NOT captured:")
print("    - COVID-19 pandemic (2020-2022)")
print("    - Multiple property cooling measures")
print("    - Interest rate changes")
print("    - Inflation and economic shifts")
print("    - Changes in HDB policy and supply")
print("\n📚 For LEARNING purposes ONLY - NOT for actual investment decisions")

print("\n✅ All done!")
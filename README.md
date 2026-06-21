# 1. On my-model.py and my-app.py
Run everything with terminal shell:
> chmod +x run.sh (not needed if already previously set to executable)

> ./run.sh

OR: Streamlit run my-app.py directly. Pickle file of model is already generated as 'house_model_extrapolated.csv'.

> streamlit run my-app.py

OR: Open the deployed streamlit app online Community:
[Streamlit deployed app hdb_price_predictor ∙ main ∙ my-app.py](https://hdb-future-price-predictor-dataset2017to19.streamlit.app/)

# 2. On my-model-final.py and my-app-final.py
Run everything with terminal shell:
> chmod +x run-final.sh (not needed if already previously set to executable)

> ./run-final.sh

OR: Streamlit run my-app-final.py directly. Pickle file of model is already generated as '/report/my-model-final/hdb_price_predictor_my-model-final.pkl'

> streamlit run my-app-final.py

OR: Open the deployed streamlit app online Community:
[Streamlit deployed app ]()

```
numerical_features = ['storey_mid', 'flat_age']  # floor_area_sqm removed
categorical_features = ['town', 'flat_category'] # flat_category (combined of flat_type and flat_model)
target = 'resale_price'
```

---
# 3. How to use the my-model-final.py

Refer notebook/use my-model-final for predict.ipynb

```
# ============================================================
# HDB PRICE PREDICTION - SINGLE PREDICTION EXAMPLE
# ============================================================

# Load the model
import joblib
import pandas as pd

# Load the model
model = joblib.load('../report/my-model-final/hdb_price_predictor_my-model-final.pkl')

# Define prediction function
def predict_price(storey_mid, flat_age, town, flat_category):
    input_data = pd.DataFrame({
        'storey_mid': [storey_mid],
        'flat_age': [flat_age],
        'town': [town],
        'flat_category': [flat_category]
    })
    return model.predict(input_data)[0]

# ============================================================
# EXAMPLE 1: 5-Room Improved Flat in Serangoon
# ============================================================

# Make prediction
price = predict_price(
    storey_mid=8,
    flat_age=25,
    town='SERANGOON',
    flat_category='5 ROOM - Improved'
)

# Print results with formatting
print("="*60)
print("🏠 HDB PRICE PREDICTION RESULT")
print("="*60)
print(f"""
📋 Property Details:
  • Town:           SERANGOON
  • Flat Category:  5 ROOM - Improved
  • Storey:         Mid-point 8 (approx. 07-09)
  • Flat Age:       25 years

💰 Predicted Resale Price: ${price:,.2f}
   (Based on 2017-2019 data - for reference only)

⚠️ Note: This is an estimate based on historical data.
    Actual prices may vary significantly.
    """)
print("="*60)
```

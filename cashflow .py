import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Cashflow_dataset.csv")

    numeric_cols = [
        'Revenue', 'COGS', 'Operating_Expenses', 'Depreciation_Amortization',
        'Change_in_Inventory', 'Accounts_Receivable', 'Accounts_Payable',
        'Taxes_Paid', 'CapEx', 'Asset_Sale_Proceeds', 'Investments_Bought',
        'Investments_Sold', 'Interest_Received', 'Debt_Raised', 'Debt_Repaid',
        'Interest_Paid', 'Equity_Issued', 'Dividends_Paid', 'Net_Cash_Flow'
    ]
    categorical_cols = ['Company_Name', 'Month']
    return data, numeric_cols, categorical_cols


# -------------------- Main App --------------------
st.title("💸 Cash Flow Prediction App")
st.write("Predict Net Cash Flow for upcoming months or new companies.")

# Load dataset
data, numeric_cols, categorical_cols = load_data()

# -------------------- Data Preprocessing --------------------
X = data.drop(columns=['Net_Cash_Flow'])
y = data['Net_Cash_Flow']

# Identify numeric columns (except target)
numeric_cols_for_scaling = [col for col in numeric_cols if col != 'Net_Cash_Flow']

scaler = StandardScaler()
X[numeric_cols_for_scaling] = scaler.fit_transform(X[numeric_cols_for_scaling])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Train Models --------------------
lr_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42, n_estimators=200)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# -------------------- Evaluate Models --------------------
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

st.subheader("Model Performance")
st.write(f"**Random Forest R² Score:** {r2_rf:.2f}")

# -------------------- User Input Section --------------------
st.header("📊 Enter Company Financials for Prediction")

user_inputs = {}
for col in numeric_cols_for_scaling:
    user_inputs[col] = st.number_input(f"Enter {col}", value=0.0)

user_inputs['Company_Name'] = st.text_input("Enter Company Name", "Aditya Infotech Ltd")
user_inputs['Month'] = st.number_input("Enter Month (1–12)", min_value=1, max_value=12, value=10)

# -------------------- Predict Button --------------------
if st.button("🔮 Predict Net Cash Flow"):
    new_data = pd.DataFrame([user_inputs])

    # Ensure all training columns exist in new_data
    for col in X.columns:
        if col not in new_data.columns:
            new_data[col] = 0

    new_data = new_data[X.columns]  # reorder columns to match training
    new_data[numeric_cols_for_scaling] = scaler.transform(new_data[numeric_cols_for_scaling])

    # Predict with Random Forest
    predicted_cashflow = rf_model.predict(new_data)[0]

    st.success(f"💰 Predicted Net Cash Flow for {user_inputs['Company_Name']}: ₹{predicted_cashflow:,.2f}")

    # -------------------- Visualization --------------------
    plt.figure(figsize=(5, 4))
    plt.bar(['Predicted'], [predicted_cashflow], color='lightgreen')
    plt.ylabel('Net Cash Flow (₹)')
    plt.title(f"Predicted Cash Flow for {user_inputs['Company_Name']}")
    st.pyplot(plt)

# -------------------- Optional Download --------------------
if st.checkbox("Show Sample Predictions"):
    test_results = X_test.copy()
    test_results['Actual'] = y_test
    test_results['Predicted_RF'] = y_pred_rf
    st.dataframe(test_results[['Actual', 'Predicted_RF']].head(10))

    csv = test_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions CSV", csv, "cashflow_predictions.csv", "text/csv")

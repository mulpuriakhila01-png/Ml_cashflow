# ===========================================
# 📊 Streamlit App: Cash Flow Prediction Model
# ===========================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

st.set_page_config(page_title="Cash Flow Prediction", layout="wide")

# =====================
# 1️⃣ LOAD DATA
# =====================
@st.cache_data
def load_data():
    data = pd.read_csv("Cashflow_dataset.csv")
    return data

try:
    data = load_data()
except FileNotFoundError:
    st.error("❌ 'Cashflow_dataset.csv' not found. Please upload it in the app directory.")
    st.stop()

st.title("💰 Company Cash Flow Prediction App")
st.write("This app analyzes company financials and predicts **Net Cash Flow** using ML models.")

st.subheader("📂 Dataset Preview")
st.dataframe(data.head())

# =====================
# 2️⃣ DATA CLEANING
# =====================
numeric_cols = [
    'Revenue', 'COGS', 'Operating_Expenses', 'Depreciation_Amortization',
    'Change_in_Inventory', 'Accounts_Receivable', 'Accounts_Payable', 'Taxes_Paid',
    'CapEx', 'Asset_Sale_Proceeds', 'Investments_Bought', 'Investments_Sold',
    'Interest_Received', 'Debt_Raised', 'Debt_Repaid', 'Interest_Paid',
    'Equity_Issued', 'Dividends_Paid', 'Net_Cash_Flow'
]
categorical_cols = ['Company_Name', 'Month']

# Fill missing values
for col in categorical_cols:
    data[col].fillna('Unknown', inplace=True)
for col in numeric_cols:
    data[col].fillna(data[col].median(), inplace=True)

# =====================
# 3️⃣ FEATURE ENGINEERING
# =====================
data['Operating_Cash_Flow'] = (
    (data['Revenue'] - data['COGS'] - data['Operating_Expenses'])
    + data['Depreciation_Amortization']
    - (data['Change_in_Inventory'] + data['Accounts_Receivable'] - data['Accounts_Payable'])
    - data['Taxes_Paid']
)

data['Investing_Cash_Flow'] = (
    (-data['CapEx'])
    + data['Asset_Sale_Proceeds']
    - data['Investments_Bought']
    + data['Investments_Sold']
    + data['Interest_Received']
)

data['Financing_Cash_Flow'] = (
    data['Debt_Raised']
    - data['Debt_Repaid']
    - data['Interest_Paid']
    + data['Equity_Issued']
    - data['Dividends_Paid']
)

data['Cash_Flow'] = (
    data['Operating_Cash_Flow']
    + data['Investing_Cash_Flow']
    + data['Financing_Cash_Flow']
)

engineered_features = ['Operating_Cash_Flow', 'Investing_Cash_Flow', 'Financing_Cash_Flow']
numeric_cols.extend(engineered_features)

# =====================
# 4️⃣ ENCODING & SCALING
# =====================
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_cols = pd.DataFrame(
    encoder.fit_transform(data[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols)
)
data_encoded = pd.concat([data.drop(columns=categorical_cols).reset_index(drop=True),
                          encoded_cols.reset_index(drop=True)], axis=1)

scaler = StandardScaler()
data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])

# =====================
# 5️⃣ TRAIN TEST SPLIT
# =====================
X = data_encoded.drop(columns=['Net_Cash_Flow'])
y = data_encoded['Net_Cash_Flow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================
# 6️⃣ MODEL TRAINING
# =====================
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    k = X_train.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    st.write(f"### {model_name} Metrics")
    st.write(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}, Adjusted R²: {adj_r2:.3f}")
    return r2

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_r2 = evaluate_model(y_test, lr_model.predict(X_test), "Linear Regression")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_r2 = evaluate_model(y_test, rf_model.predict(X_test), "Random Forest")

# XGBoost
xgb_model = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_r2 = evaluate_model(y_test, xgb_model.predict(X_test), "XGBoost")

# Model comparison chart
model_scores = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'R² Score': [lr_r2, rf_r2, xgb_r2]
})
st.bar_chart(model_scores.set_index('Model'))

# =====================
# 7️⃣ USER INPUT PREDICTION
# =====================
st.subheader("🔮 Predict Net Cash Flow for a New Company")

company_name = st.text_input("Company Name", "ABC Ltd")
month = st.selectbox("Month", [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])

revenue = st.number_input("Revenue", min_value=0.0)
cogs = st.number_input("COGS", min_value=0.0)
op_exp = st.number_input("Operating Expenses", min_value=0.0)
dep_amort = st.number_input("Depreciation & Amortization", min_value=0.0)
change_inv = st.number_input("Change in Inventory", value=0.0)
acc_recv = st.number_input("Accounts Receivable", value=0.0)
acc_pay = st.number_input("Accounts Payable", value=0.0)
taxes_paid = st.number_input("Taxes Paid", value=0.0)
capex = st.number_input("CapEx", value=0.0)
asset_sale = st.number_input("Asset Sale Proceeds", value=0.0)
invest_bought = st.number_input("Investments Bought", value=0.0)
invest_sold = st.number_input("Investments Sold", value=0.0)
interest_recv = st.number_input("Interest Received", value=0.0)
debt_raised = st.number_input("Debt Raised", value=0.0)
debt_repaid = st.number_input("Debt Repaid", value=0.0)
interest_paid = st.number_input("Interest Paid", value=0.0)
equity_issued = st.number_input("Equity Issued", value=0.0)
dividends_paid = st.number_input("Dividends Paid", value=0.0)

if st.button("🚀 Predict Cash Flow"):
    operating_cf = (revenue - cogs - op_exp) + dep_amort - (change_inv + acc_recv - acc_pay) - taxes_paid
    investing_cf = (-capex) + asset_sale - invest_bought + invest_sold + interest_recv
    financing_cf = debt_raised - debt_repaid - interest_paid + equity_issued - dividends_paid

    new_data = pd.DataFrame({
        'Revenue': [revenue],
        'COGS': [cogs],
        'Operating_Expenses': [op_exp],
        'Depreciation_Amortization': [dep_amort],
        'Change_in_Inventory': [change_inv],
        'Accounts_Receivable': [acc_recv],
        'Accounts_Payable': [acc_pay],
        'Taxes_Paid': [taxes_paid],
        'CapEx': [capex],
        'Asset_Sale_Proceeds': [asset_sale],
        'Investments_Bought': [invest_bought],
        'Investments_Sold': [invest_sold],
        'Interest_Received': [interest_recv],
        'Debt_Raised': [debt_raised],
        'Debt_Repaid': [debt_repaid],
        'Interest_Paid': [interest_paid],
        'Equity_Issued': [equity_issued],
        'Dividends_Paid': [dividends_paid],
        'Operating_Cash_Flow': [operating_cf],
        'Investing_Cash_Flow': [investing_cf],
        'Financing_Cash_Flow': [financing_cf]
    })

    new_data = new_data.reindex(columns=X_train.columns, fill_value=0)
    new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    predicted_cf = xgb_model.predict(new_data)[0]

    st.success(f"💰 Predicted Net Cash Flow: ₹{predicted_cf:,.2f}")
    st.write("### Cash Flow Breakdown")
    st.dataframe(pd.DataFrame({
        "Operating CF": [operating_cf],
        "Investing CF": [investing_cf],
        "Financing CF": [financing_cf],
        "Predicted Net CF": [predicted_cf]
    }).style.format("{:,.2f}"))

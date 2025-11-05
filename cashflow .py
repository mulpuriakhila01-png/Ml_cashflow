import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# -----------------------------
# STREAMLIT APP CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Cash Flow Prediction App", layout="wide")
st.title("💰 Cash Flow Prediction with ML Models")

# -----------------------------
# UPLOAD DATA
# -----------------------------
uploaded_file = st.file_uploader("Upload your Cashflow Dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    st.write("Dataset Shape:", data.shape)

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    numeric_cols = [
        'Revenue', 'COGS', 'Operating_Expenses', 'Depreciation_Amortization',
        'Change_in_Inventory', 'Accounts_Receivable', 'Accounts_Payable', 'Taxes_Paid',
        'CapEx', 'Asset_Sale_Proceeds', 'Investments_Bought', 'Investments_Sold',
        'Interest_Received', 'Debt_Raised', 'Debt_Repaid', 'Interest_Paid',
        'Equity_Issued', 'Dividends_Paid', 'Net_Cash_Flow'
    ]
    categorical_cols = ['Company_Name', 'Month']

    for col in categorical_cols:
        data[col].fillna('Unknown', inplace=True)
    for col in numeric_cols:
        data[col].fillna(data[col].median(), inplace=True)

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
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

    st.subheader("🧮 Feature Engineered Columns")
    st.dataframe(data[['Company_Name','Month','Operating_Cash_Flow','Investing_Cash_Flow','Financing_Cash_Flow','Cash_Flow']].head())

    # -----------------------------
    # CORRELATION HEATMAP
    # -----------------------------
    st.subheader("🔍 Correlation Heatmap")
    plt.figure(figsize=(12,6))
    sns.heatmap(data[numeric_cols + ['Operating_Cash_Flow','Investing_Cash_Flow','Financing_Cash_Flow']].corr(), annot=False, cmap="coolwarm")
    st.pyplot(plt)

    # -----------------------------
    # OUTLIER TREATMENT (IQR)
    # -----------------------------
    def cap_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[column] = np.where(df[column] < lower, lower,
                              np.where(df[column] > upper, upper, df[column]))
        return df

    for col in numeric_cols:
        data = cap_outliers_iqr(data, col)

    # -----------------------------
    # ENCODING & SCALING
    # -----------------------------
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]),
                                columns=encoder.get_feature_names_out(categorical_cols))
    data_encoded = pd.concat([data.drop(columns=categorical_cols).reset_index(drop=True),
                              encoded_cols.reset_index(drop=True)], axis=1)

    scaler = StandardScaler()
    numeric_features = numeric_cols + ['Operating_Cash_Flow', 'Investing_Cash_Flow', 'Financing_Cash_Flow']
    data_encoded[numeric_features] = scaler.fit_transform(data_encoded[numeric_features])

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    X = data_encoded.drop(columns=['Net_Cash_Flow'])
    y = data_encoded['Net_Cash_Flow']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression().fit(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ).fit(X_train, y_train)

    # -----------------------------
    # EVALUATION FUNCTION
    # -----------------------------
    def evaluate(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R²": r2_score(y_true, y_pred)
        }

    lr_res = evaluate(y_test, lr_model.predict(X_test))
    rf_res = evaluate(y_test, rf_model.predict(X_test))
    xgb_res = evaluate(y_test, xgb_model.predict(X_test))

    results = pd.DataFrame([lr_res, rf_res, xgb_res], index=['Linear Regression','Random Forest','XGBoost'])
    st.subheader("📈 Model Performance Comparison")
    st.dataframe(results)

    st.bar_chart(results['R²'])

    # -----------------------------
    # USER INPUT FORM FOR PREDICTION
    # -----------------------------
    st.header("🧮 Predict Net Cash Flow")

    with st.form("prediction_form"):
        st.write("Enter the financial details below:")
        revenue = st.number_input("Revenue")
        cogs = st.number_input("COGS")
        op_exp = st.number_input("Operating Expenses")
        dep = st.number_input("Depreciation & Amortization")
        inv_change = st.number_input("Change in Inventory")
        acc_recv = st.number_input("Accounts Receivable")
        acc_pay = st.number_input("Accounts Payable")
        taxes = st.number_input("Taxes Paid")
        capex = st.number_input("CapEx")
        asset_sale = st.number_input("Asset Sale Proceeds")
        inv_bought = st.number_input("Investments Bought")
        inv_sold = st.number_input("Investments Sold")
        int_recv = st.number_input("Interest Received")
        debt_raised = st.number_input("Debt Raised")
        debt_repaid = st.number_input("Debt Repaid")
        int_paid = st.number_input("Interest Paid")
        equity_issued = st.number_input("Equity Issued")
        div_paid = st.number_input("Dividends Paid")

        submitted = st.form_submit_button("Predict")

        if submitted:
            op_cf = (revenue - cogs - op_exp) + dep - (inv_change + acc_recv - acc_pay) - taxes
            inv_cf = (-capex) + asset_sale - inv_bought + inv_sold + int_recv
            fin_cf = debt_raised - debt_repaid - int_paid + equity_issued - div_paid
            total_cf = op_cf + inv_cf + fin_cf

            st.write(f"**Operating Cash Flow:** ₹{op_cf:,.2f}")
            st.write(f"**Investing Cash Flow:** ₹{inv_cf:,.2f}")
            st.write(f"**Financing Cash Flow:** ₹{fin_cf:,.2f}")
            st.write(f"**Predicted Net Cash Flow:** ₹{total_cf:,.2f}")
else:
    st.warning("👆 Please upload a CSV file to continue.")

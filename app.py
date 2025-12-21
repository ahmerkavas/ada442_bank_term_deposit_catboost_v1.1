import streamlit as st
import pandas as pd
import joblib

# Load CatBoost model (duration removed version)
model = joblib.load("catboost_model.joblib")

# Feature list WITHOUT "duration"
FEATURES = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]

st.title("Bank Term Deposit Prediction (CatBoost v1.1)")
st.write("Predict whether a customer will subscribe to a term deposit.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    job = st.selectbox("Job", [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ])

    marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
    education = st.selectbox("Education", [
        "basic.4y", "basic.6y", "basic.9y",
        "high.school", "illiterate",
        "professional.course", "university.degree", "unknown"
    ])

    default = st.selectbox("Default", ["no", "yes", "unknown"])
    housing = st.selectbox("Housing Loan", ["no", "yes", "unknown"])
    loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])

    month = st.selectbox("Month", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ])
    day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])

    campaign = st.number_input("Number of Contacts (Campaign)", min_value=1, value=1)
    pdays = st.number_input("Days Since Last Contact", min_value=0, value=999)
    previous = st.number_input("Previous Contacts", min_value=0, value=0)
    poutcome = st.selectbox("Previous Outcome", ["failure", "success", "nonexistent"])

    emp_var_rate = st.number_input("Employment Variation Rate (emp.var.rate)", value=1.1)
    cons_price_idx = st.number_input("Consumer Price Index (cons.price.idx)", value=93.994)
    cons_conf_idx = st.number_input("Consumer Confidence Index (cons.conf.idx)", value=-36.4)
    euribor3m = st.number_input("Euribor 3 Month Rate (euribor3m)", value=4.855)
    nr_employed = st.number_input("Number of Employed (nr.employed)", value=5191.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed
    }

    X = pd.DataFrame([row], columns=FEATURES)

    # Predict label
    pred = model.predict(X)
    pred_value = pred[0] if hasattr(pred, "__len__") else pred

    # Predict probability (robust)
    prob_yes = None
    try:
        proba = model.predict_proba(X)
        if hasattr(proba, "shape") and len(proba.shape) == 2 and proba.shape[1] >= 2:
            prob_yes = float(proba[0][1])
        else:
            prob_yes = float(proba[0])
    except Exception:
        prob_yes = None

    # Normalize prediction to YES/NO
    pred_str = str(pred_value).strip().lower()
    is_yes = (pred_value == 1) or (pred_str in {"1", "yes", "true", "y"})

    if prob_yes is not None:
        if is_yes:
            st.success(f"Prediction: YES (Probability: {prob_yes:.2f})")
        else:
            st.error(f"Prediction: NO (Probability: {prob_yes:.2f})")
    else:
        if is_yes:
            st.success("Prediction: YES")
        else:
            st.error("Prediction: NO")

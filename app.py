import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load trained model
model = joblib.load("../models/fraud_model.pkl")

st.title("Fraud Detection App")

#  1. load CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
#  2. Choose columns
    features = ['transaction_amount', 'customer_age', 'customer_balance']
    X = df[features]
    df['is_fraud_prediction'] = model.predict(X)
# 3 Add transaction ratio columns
    df['transaction_ratio'] = df['transaction_amount'] / df['customer_balance']
# 4 Filtered by fraud
    st.subheader("Preview of Predictions")
    fraud_filter = st.radio(
        "Show transactions:",
        ("All", "Only Fraudulent", "Only Non-Fraudulent")
    )

    if fraud_filter == "Only Fraudulent":
        df = df[df['is_fraud_prediction'] == 1]
    elif fraud_filter == "Only Non-Fraudulent":
        df = df[df['is_fraud_prediction'] == 0]
# 5 Reoder columns
    df = df[["transaction_amount", "customer_age", "customer_balance", "transaction_ratio", "is_fraud_prediction"]]
# 6 Fraud summary
    num_fraud = df['is_fraud_prediction'].sum()
    total = len(df)
    fraud_percent = (num_fraud / total) * 100 if total > 0 else 0

    st.markdown(f"### Fraud Summary")
    st.markdown(f"- **Total transactions:** {total}")
    st.markdown(f"- **Fraudulent transactions:** {int(num_fraud)}")
    st.markdown(f"- **Fraud rate:** {fraud_percent:.2f}%")
# 7 Styled data table
    styled_df = df.style \
        .background_gradient(subset=["transaction_ratio"], cmap="Blues") \
        .highlight_max(color="lightgreen", axis=0, subset=["transaction_amount", "customer_balance"]) \
        .format({"transaction_ratio": "{:.2f}"})
    st.dataframe(styled_df, use_container_width=True)

# 8 interactive scatter plot
    st.subheader("Explore Data with Scatter Plot")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    x_col = st.selectbox("Select X-axis", numeric_cols, index=0)
    y_col = st.selectbox("Select Y-axis", numeric_cols, index=1)

    fig = px.scatter(df, x=x_col, y=y_col, color='is_fraud_prediction',
                        title=f"{y_col} vs {x_col}", labels={"color": "Fraud?"})
    st.plotly_chart(fig)

# 9 Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
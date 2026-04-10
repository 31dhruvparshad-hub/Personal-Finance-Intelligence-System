import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Finance AI Dashboard", layout="wide")

# ------------------ HEADER ------------------
st.title("💰 Finance Intelligence Dashboard")
st.caption("Smart insights. Better financial decisions.")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload your financial dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ------------------ CLEANING ------------------
    df.columns = ['date', 'description', 'category', 'amount', 'flow']
    df['date'] = pd.to_datetime(df['date'])
    df['category'] = df['category'].str.lower().str.strip()
    df['flow'] = df['flow'].str.lower().str.strip()
    df['month'] = df['date'].dt.month

    df_expense = df[df['flow'] == 'expense']
    df_income = df[df['flow'] == 'income']

    # ------------------ KPI CARDS ------------------
    total_expense = df_expense['amount'].sum()
    total_income = df_income['amount'].sum()
    savings = total_income - total_expense

    col1, col2, col3 = st.columns(3)

    col1.metric("💸 Total Expense", f"₹ {total_expense:,.0f}")
    col2.metric("💰 Total Income", f"₹ {total_income:,.0f}")
    col3.metric("📈 Savings", f"₹ {savings:,.0f}")

    st.markdown("---")

    # ------------------ CHARTS ------------------
    col1, col2 = st.columns(2)

    # CATEGORY PIE CHART
    with col1:
        st.subheader("📊 Spending Breakdown")

        category_spend = df_expense.groupby('category')['amount'].sum().reset_index()

        fig = px.pie(
            category_spend,
            names='category',
            values='amount',
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

    # MONTHLY TREND
    with col2:
        st.subheader("📈 Monthly Expense Trend")

        monthly = df_expense.groupby('month')['amount'].sum().reset_index()

        fig = px.line(
            monthly,
            x='month',
            y='amount',
            markers=True
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------ ANOMALY DETECTION ------------------
    st.subheader("🚨 Spending Alerts")

    mean = monthly['amount'].mean()
    std = monthly['amount'].std()

    threshold_high = mean + 1.5 * std
    threshold_low = mean - 1.5 * std

    anomalies = monthly[
        (monthly['amount'] > threshold_high) |
        (monthly['amount'] < threshold_low)
    ]

    if not anomalies.empty:
        st.error(f"⚠️ Unusual spending in month(s): {list(anomalies['month'])}")
    else:
        st.success("✅ Spending pattern is stable")

    st.markdown("---")

    # ------------------ PREDICTION ------------------
    st.subheader("🔮 AI Prediction")

    X = monthly[['month']]
    y = monthly['amount']

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict([[13]])

    st.metric("Next Month Expected Expense", f"₹ {prediction[0]:,.0f}")

    st.markdown("---")

    # ------------------ INSIGHTS ------------------
    st.subheader("💡 Smart Insights")

    top_category = category_spend.loc[
        category_spend['amount'].idxmax(), 'category'
    ]

    percentage = (
        category_spend['amount'].max() /
        category_spend['amount'].sum()
    ) * 100

    st.write(f"🔹 Highest spending: **{top_category} ({percentage:.1f}%)**")

    if percentage > 30:
        st.warning("⚠️ You are spending heavily in one category")

    if savings < 0:
        st.error("🚨 You are overspending!")
    else:
        st.success("✅ You are saving money")

else:
    
    # HERO SECTION
    st.markdown("""
    <h1 style='text-align: center;'>💰 Finance Intelligence System</h1>
    <p style='text-align: center; font-size:18px;'>
    Take control of your money with AI-powered insights
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # HERO IMAGE
    st.image(
        "https://images.unsplash.com/photo-1554224155-6726b3ff858f",
        use_container_width=True
    )

    st.markdown("---")

    # FEATURES SECTION
    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    ### 📊 Analyze Spending  
    Understand where your money goes  
    """)

    col2.markdown("""
    ### 📈 Track Trends  
    Monitor monthly patterns  
    """)

    col3.markdown("""
    ### 🤖 AI Predictions  
    Forecast future expenses  
    """)

    st.markdown("---")

    # CALL TO ACTION
    st.success("⬆️ Upload your dataset above to get started")
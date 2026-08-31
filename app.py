import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import date

# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="SmartCart | Customer Segmentation",
    page_icon="🛒",
    layout="centered",
)

# -----------------------------------------------------------------
# Load custom CSS (won't crash the app if missing)
# -----------------------------------------------------------------
def load_css(file_path: str):
    css_path = Path(__file__).parent / file_path
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Could not find {file_path} — running with default styling.")

load_css("style.css")

# -----------------------------------------------------------------
# Load model artifacts (cached so they load only once)
# -----------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent
    scaler = joblib.load(base / "scaler.joblib")
    ohe = joblib.load(base / "ohe.joblib")
    pca = joblib.load(base / "pca.joblib")
    kmeans = joblib.load(base / "kmeans.joblib")
    feature_columns = joblib.load(base / "feature_columns.joblib")
    cat_cols = joblib.load(base / "cat_cols.joblib")
    reference_date = joblib.load(base / "reference_date.joblib")
    cluster_summary = joblib.load(base / "cluster_summary.joblib")
    segment_names = joblib.load(base / "segment_names.joblib")
    return scaler, ohe, pca, kmeans, feature_columns, cat_cols, reference_date, cluster_summary, segment_names

(scaler, ohe, pca, kmeans, feature_columns, cat_cols,
 reference_date, cluster_summary, segment_names) = load_artifacts()

# -----------------------------------------------------------------
# Header
# -----------------------------------------------------------------
st.markdown(
    """
    <div class="sc-header">
        <h1>🛒 SmartCart</h1>
        <p>Discover which customer segment a shopper belongs to — powered by PCA + K-Means clustering.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------
# Input form
# -----------------------------------------------------------------
with st.form("segmentation_form"):

    st.markdown('<div class="sc-card"><h3>👤 Demographics</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        birth_year = st.number_input("Year of Birth", min_value=1930, max_value=2010, value=1985)
        education = st.selectbox("Education", ["Undergraduate", "Graduate", "Postgraduate"])
    with col2:
        living_with = st.selectbox("Living Situation", ["Partner", "Alone"])
        income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    col1, col2 = st.columns(2)
    with col1:
        kidhome = st.number_input("Kids at Home", min_value=0, max_value=5, value=0)
    with col2:
        teenhome = st.number_input("Teens at Home", min_value=0, max_value=5, value=0)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sc-card"><h3>🛍️ Spending (last 2 years)</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        wines = st.number_input("Wines ($)", min_value=0, value=100, step=10)
        fruits = st.number_input("Fruits ($)", min_value=0, value=20, step=5)
    with col2:
        meat = st.number_input("Meat ($)", min_value=0, value=150, step=10)
        fish = st.number_input("Fish ($)", min_value=0, value=30, step=5)
    with col3:
        sweets = st.number_input("Sweets ($)", min_value=0, value=20, step=5)
        gold = st.number_input("Gold Products ($)", min_value=0, value=40, step=5)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sc-card"><h3>📈 Engagement & Purchase Behavior</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        recency = st.number_input("Days Since Last Purchase (Recency)", min_value=0, max_value=365, value=30)
        deals = st.number_input("Purchases Using Deals", min_value=0, max_value=20, value=2)
        web_purchases = st.number_input("Web Purchases", min_value=0, max_value=30, value=4)
    with col2:
        catalog_purchases = st.number_input("Catalog Purchases", min_value=0, max_value=30, value=2)
        store_purchases = st.number_input("Store Purchases", min_value=0, max_value=30, value=5)
        web_visits = st.number_input("Web Visits per Month", min_value=0, max_value=30, value=5)
    col1, col2 = st.columns(2)
    with col1:
        complain = st.selectbox("Filed a Complaint?", ["No", "Yes"])
    with col2:
        response = st.selectbox("Responded to Last Campaign?", ["No", "Yes"])
    join_date = st.date_input("Customer Since", value=date(2013, 1, 1))
    st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Find My Segment")

# -----------------------------------------------------------------
# Prediction logic
# -----------------------------------------------------------------
if submitted:
    age = 2026 - birth_year
    tenure_days = (pd.Timestamp(reference_date) - pd.Timestamp(join_date)).days
    total_spending = wines + fruits + meat + fish + sweets + gold
    total_children = kidhome + teenhome

    raw_input = pd.DataFrame([{
        "Income": income,
        "Recency": recency,
        "NumDealsPurchases": deals,
        "NumWebPurchases": web_purchases,
        "NumCatalogPurchases": catalog_purchases,
        "NumStorePurchases": store_purchases,
        "NumWebVisitsMonth": web_visits,
        "Complain": 1 if complain == "Yes" else 0,
        "Response": 1 if response == "Yes" else 0,
        "Age": age,
        "Customer_Tenure_Days": tenure_days,
        "Total_Spending": total_spending,
        "Total_Children": total_children,
        "Education": education,
        "Living_With": living_with,
    }])

    # One-hot encode categorical columns the same way as training
    encoded = ohe.transform(raw_input[cat_cols])
    encoded_df = pd.DataFrame(
        encoded.toarray(), columns=ohe.get_feature_names_out(cat_cols), index=raw_input.index
    )
    processed = pd.concat([raw_input.drop(columns=cat_cols), encoded_df], axis=1)

    # Align to the exact training column order
    processed = processed.reindex(columns=feature_columns, fill_value=0)

    # Scale -> PCA -> predict cluster
    scaled_input = scaler.transform(processed)
    pca_input = pca.transform(scaled_input)
    cluster = kmeans.predict(pca_input)[0]

    segment_name = segment_names.get(int(cluster), f"Segment {cluster}")
    profile = cluster_summary.loc[cluster]

    st.markdown(
        f"""
        <div class="sc-result">
            <h2>🎯 {segment_name}</h2>
            <span class="sc-badge">Cluster {cluster}</span>
            <p style="margin-top:1rem;">This customer's spending and engagement pattern most closely matches
            other shoppers in this segment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📊 How this segment compares (average values)")
    compare_df = pd.DataFrame({
        "This Segment's Average": [
            f"${profile['Income']:,.0f}", f"${profile['Total_Spending']:,.0f}",
            f"{profile['Total_Children']:.1f}", f"{profile['Age']:.0f} yrs",
            f"{profile['Recency']:.0f} days",
        ],
        "This Customer": [
            f"${income:,.0f}", f"${total_spending:,.0f}",
            f"{total_children}", f"{age} yrs", f"{recency} days",
        ],
    }, index=["Income", "Total Spending", "Children at Home", "Age", "Recency"])
    st.table(compare_df)

    st.caption(
        "⚠️ Segments are derived from unsupervised clustering on a sample marketing dataset, "
        "for educational purposes only."
    )

# -----------------------------------------------------------------
# Footer
# -----------------------------------------------------------------
st.markdown(
    '<div class="sc-footer">Built with 💜 using Streamlit & Scikit-learn — SmartCart Project</div>',
    unsafe_allow_html=True,
)

"""
Trains the SmartCart customer segmentation pipeline, replicating the
preprocessing + clustering steps from the original notebook, and saves
all artifacts needed by the Streamlit app.

Note: The notebook compared K-Means and Agglomerative Clustering for
characterizing the 4 segments. For the live app, we use K-Means because
it has a .predict() method that can assign a *new* customer to a cluster
without needing to refit on the whole dataset - Agglomerative Clustering
cannot do this.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("smartcart_customers.csv")

# ---------------------------------------------------------------
# 2. Handle missing values
# ---------------------------------------------------------------
df["Income"] = df["Income"].fillna(df["Income"].median())

# ---------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------
CURRENT_YEAR = 2026
df["Age"] = CURRENT_YEAR - df["Year_Birth"]

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
reference_date = df["Dt_Customer"].max()
df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

df["Total_Spending"] = (
    df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
    + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
)

df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

df["Education"] = df["Education"].replace({
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate",
    "Master": "Postgraduate", "PhD": "Postgraduate"
})

df["Living_With"] = df["Marital_Status"].replace({
    "Married": "Partner", "Together": "Partner",
    "Single": "Alone", "Divorced": "Alone", "Widow": "Alone",
    "Absurd": "Alone", "YOLO": "Alone"
})

# ---------------------------------------------------------------
# 4. Drop unneeded columns
# ---------------------------------------------------------------
cols = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"]
spending_cols = ["MntWines", "MntFruits", "MntMeatProducts",
                  "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
cols_to_drop = cols + spending_cols
df_cleaned = df.drop(columns=cols_to_drop)

# ---------------------------------------------------------------
# 5. Remove outliers
# ---------------------------------------------------------------
df_cleaned = df_cleaned[df_cleaned["Age"] < 90]
df_cleaned = df_cleaned[df["Income"] < 600_000]

# ---------------------------------------------------------------
# 6. Encoding
# ---------------------------------------------------------------
cat_cols = ["Education", "Living_With"]
ohe = OneHotEncoder(handle_unknown="ignore")
enc_cols = ohe.fit_transform(df_cleaned[cat_cols])
enc_df = pd.DataFrame(
    enc_cols.toarray(), columns=ohe.get_feature_names_out(cat_cols), index=df_cleaned.index
)
df_encoded = pd.concat([df_cleaned.drop(columns=cat_cols), enc_df], axis=1)

feature_columns = list(df_encoded.columns)  # exact column order expected downstream

# ---------------------------------------------------------------
# 7. Scaling
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# ---------------------------------------------------------------
# 8. PCA (3 components, matches the notebook's 3D clustering)
# ---------------------------------------------------------------
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ---------------------------------------------------------------
# 9. K-Means clustering (k=4, chosen via elbow + silhouette in the notebook)
# ---------------------------------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_pca)

# ---------------------------------------------------------------
# 10. Build human-readable cluster profiles
# ---------------------------------------------------------------
df_encoded["cluster"] = labels
profile_cols = ["Income", "Total_Spending", "Total_Children", "Age",
                 "Recency", "Living_With_Partner", "Living_With_Alone"]
cluster_summary = df_encoded.groupby("cluster")[profile_cols].mean().round(1)
print("Cluster profiles:\n", cluster_summary)

# Rank clusters by spending tier (highest to lowest) for a clear, unique label per cluster
spend_ranked = cluster_summary["Total_Spending"].sort_values(ascending=False).index.tolist()
tier_labels = ["Premium Spenders", "Established Spenders", "Emerging Spenders", "Budget-Conscious Shoppers"]
tier_map = {cluster_id: tier_labels[i] for i, cluster_id in enumerate(spend_ranked[:len(tier_labels)])}

segment_names = {}
for c in cluster_summary.index:
    living_partner = cluster_summary.loc[c, "Living_With_Partner"]
    household = "Partnered Households" if living_partner >= 0.5 else "Independent Households"
    segment_names[int(c)] = f"{tier_map[c]} · {household}"

print("\nSegment names:", segment_names)

# ---------------------------------------------------------------
# 11. Save all artifacts the app needs
# ---------------------------------------------------------------
joblib.dump(scaler, "scaler.joblib")
joblib.dump(ohe, "ohe.joblib")
joblib.dump(pca, "pca.joblib")
joblib.dump(kmeans, "kmeans.joblib")
joblib.dump(feature_columns, "feature_columns.joblib")
joblib.dump(cat_cols, "cat_cols.joblib")
joblib.dump(reference_date, "reference_date.joblib")
joblib.dump(cluster_summary, "cluster_summary.joblib")
joblib.dump(segment_names, "segment_names.joblib")

print("\nAll artifacts saved successfully.")

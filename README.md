# 🛒 SmartCart — Customer Segmentation (Streamlit App)

A live, interactive web app that assigns a shopper to one of four customer segments using PCA + K-Means clustering. This is the deployable companion to the [SmartCart ML notebook project](#) — same pipeline, same clusters, wrapped in a simple UI.

🔗 **Live demo:** `<add your Streamlit Cloud link here after deployment>`

## 🖥️ What it does

Fill in a customer's demographics, spending, and engagement behavior, and the app assigns them to one of four discovered segments:

- **Premium Spenders** — high income, high spending households
- **Established Spenders** — comfortably high income and spending
- **Emerging Spenders** — moderate spending, typically independent households
- **Budget-Conscious Shoppers** — lower income and spending, often with children

It also shows how the entered customer compares to the segment's average profile.

## 📁 Files in this repo

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (UI + clustering logic) |
| `style.css` | Custom styling for the app |
| `train_model.py` | Script that reproduces the pipeline from raw data |
| `smartcart_customers.csv` | Training dataset |
| `scaler.joblib` | StandardScaler fitted on training data |
| `ohe.joblib` | OneHotEncoder fitted on Education & Living_With |
| `pca.joblib` | PCA (3 components) fitted on scaled training data |
| `kmeans.joblib` | Trained K-Means model (k=4) used to assign new customers to a segment |
| `feature_columns.joblib` | Column schema used to align new inputs at inference time |
| `cat_cols.joblib` | Names of the categorical columns that get one-hot encoded |
| `reference_date.joblib` | The reference date used to compute customer tenure, kept consistent between training and inference |
| `cluster_summary.joblib` | Average profile (income, spending, etc.) per segment, used for the comparison table |
| `segment_names.joblib` | Human-readable names mapped to each cluster number |
| `requirements.txt` | Python dependencies |

## 🚀 Run locally

```bash
git clone https://github.com/<your-username>/smartcart-streamlit-app.git
cd smartcart-streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 🔁 Retrain the model (optional)

```bash
python train_model.py
```

This reruns the full preprocessing, PCA, and K-Means pipeline and overwrites the `.joblib` files.

## ℹ️ Why K-Means and not Agglomerative Clustering?

The original notebook compared both K-Means and Agglomerative Clustering to characterize the four segments. For this live app, K-Means is used because it has a `.predict()` method that can assign a **brand-new** customer to a cluster instantly. Agglomerative Clustering has no such method — it can only cluster the exact dataset it was fit on, which makes it unsuitable for real-time predictions on new inputs.

## 🛠️ Tech Stack

Streamlit · Scikit-learn · Pandas · NumPy

## ⚠️ Disclaimer

This app is for educational purposes only, based on a sample marketing dataset.

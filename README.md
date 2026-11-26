# Customer Segmentation (PCA + K-Means)

An unsupervised learning project that segments retail customers by income, spending and purchasing behaviour, using PCA for dimensionality reduction and K-Means and Agglomerative clustering. The results are shown in a Streamlit app.

**Live app:** [LIVE-LINK](https://smartcart-app-app-dfejdjwfubjzz59hftsxtz.streamlit.app/)

<img width="1133" height="689" alt="image" src="https://github.com/user-attachments/assets/566f5806-f101-438f-87c0-b11ce7975423" />
<img width="1162" height="776" alt="image" src="https://github.com/user-attachments/assets/d7fc81f8-cdde-44be-8624-c119f7eaf84d" />
<img width="1021" height="767" alt="image" src="https://github.com/user-attachments/assets/9baa4040-8b3e-48df-bd82-e3ecab39fe5f" />

## Dataset

2,240 customers with 22 columns (demographics, income, product spending, purchase channels, campaign response). Source: [add dataset source and link].

## Approach

1. **Missing values:** filled missing income with the median.
2. **Feature engineering:** created age, customer tenure (days), total spending, total children, and a "living with" feature; grouped education into three levels.
3. **Outliers:** removed ages of 90 and above and incomes of 600,000 and above, leaving 2,236 customers.
4. **Encoding and scaling:** one-hot encoded categorical columns (18 features in total) and standardized them.
5. **PCA:** reduced to 3 components, which explain 45% of the variance.
6. **Choosing K:** elbow method (with KneeLocator) gave K = 4; silhouette scores were also checked for K = 2 to 10.
7. **Clustering:** K-Means and Agglomerative (Ward) clustering with 4 clusters.

## Results

| Method | Clusters | Silhouette score |
|---|---|---|
| K-Means | 4 | 0.36 |
| Agglomerative (Ward) | 4 | 0.38 |

Segment profile (Agglomerative clusters):

| Cluster | Customers | Avg income | Avg total spending | Avg children |
|---|---|---|---|---|
| 0 | 905 | ~39.7K | ~222 | ~1 |
| 1 | 534 | ~72.8K | ~1,237 | ~1 |
| 2 | 444 | ~37.0K | ~166 | ~1 |
| 3 | 353 | ~70.7K | ~1,190 | 0 |

Two segments are higher-income, high-spend customers (clusters 1 and 3), and two are lower-income, low-spend customers (clusters 0 and 2). Targeted offers could focus on the high-spend groups, and the low-spend groups are candidates for engagement campaigns.

## Limitations and Next Steps

- Silhouette scores are modest, so the segments overlap; scores are higher for K of 5 and above, but K = 4 was chosen for interpretability
- PCA keeps 45% of the variance, so some information is lost
- Try other clustering methods (DBSCAN, Gaussian mixtures) and validate segment stability

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, KneeLocator, Streamlit

## Run

```bash
pip install -r requirements.txt
jupyter notebook smartcart.ipynb
streamlit run app.py
```

## Author

**Daman Dhillon**
[LinkedIn](https://www.linkedin.com/in/daman-jeet-kaur-dhillon-324191312/) | [GitHub](https://github.com/DamanDhillon16)

Copyright (c) 2026 Daman Dhillon. All rights reserved.

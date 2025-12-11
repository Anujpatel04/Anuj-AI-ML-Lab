import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------------------
# Load Dataset
# ---------------------------
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Using 2 features
X = df[['sepal length (cm)', 'sepal width (cm)']]

# ---------------------------
# Scaling the data
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# DBSCAN Model
# ---------------------------
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

df['DBSCAN_Cluster'] = labels

print("Unique Cluster Labels:", set(labels))

# Silhouette Score (valid only if more than 1 cluster)
if len(set(labels)) > 1 and -1 not in set(labels):
    print("Silhouette Score:", silhouette_score(X_scaled, labels))
else:
    print("Silhouette Score: Not applicable (noise or single cluster)")

# ---------------------------
# Plotting DBSCAN
# ---------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='plasma', s=60)
plt.title("DBSCAN Clustering")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

# Preview dataframe
print("\nDataFrame with DBSCAN labels:")
print(df.head())

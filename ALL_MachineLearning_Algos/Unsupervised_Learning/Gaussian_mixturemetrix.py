import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
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
# Gaussian Mixture Model (GMM)
# ---------------------------
gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(X_scaled)

df['GMM_Cluster'] = labels

print("Unique Cluster Labels:", set(labels))
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# ---------------------------
# Plotting GMM Clusters
# ---------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='cool', s=60)
plt.title("Gaussian Mixture Model (GMM) Clustering")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

# Preview dataframe
print("\nDataFrame with GMM labels:")
print(df.head())

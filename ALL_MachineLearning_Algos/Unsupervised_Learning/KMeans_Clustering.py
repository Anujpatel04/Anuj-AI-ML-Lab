import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# We will cluster using 2 features for visualization
X = df[['sepal length (cm)', 'sepal width (cm)']]

# Scale the data (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Predictions
labels = kmeans.labels_

# Evaluation
print("\nK-Means Clustering Metrics:")
print("Inertia (WCSS):", kmeans.inertia_)
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# Add cluster labels to DataFrame
df['Cluster'] = labels

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centers')

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("K-Means Clustering on Iris Dataset (2 Features)")
plt.legend()
plt.show()

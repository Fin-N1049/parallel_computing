import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Define paths
zip_path = "/content/archive.zip"
extract_path = "/mnt/data/extracted_files"

# Extract ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Identify the extracted CSV file
extracted_files = os.listdir(extract_path)
csv_file = None
for file in extracted_files:
    if file.endswith(".csv"):
        csv_file = os.path.join(extract_path, file)
        break

if not csv_file:
    raise FileNotFoundError("No CSV file found in the extracted ZIP archive.")

# Load dataset
supermarket_data = pd.read_csv(csv_file)
print(supermarket_data.head())  # Display first few rows

# Selecting relevant numerical features for clustering
features = ["Unit price", "Quantity", "Total", "cogs", "gross income"]

# Ensure selected columns exist
supermarket_data = supermarket_data[[col for col in features if col in supermarket_data.columns]]

# Convert to numeric and handle errors
supermarket_data = supermarket_data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
supermarket_data.dropna(inplace=True)

# Standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(supermarket_data)

# Determine optimal clusters using Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# Apply K-Means with optimal clusters (assuming 3 for this example)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
supermarket_data["Cluster"] = kmeans.fit_predict(data_scaled)

# Analyze cluster characteristics
print(supermarket_data.groupby("Cluster").mean(numeric_only=True))

# Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Scatter plot of clusters
plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=supermarket_data["Cluster"], cmap='viridis', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segments Visualization")
plt.colorbar(label='Cluster')
plt.show()

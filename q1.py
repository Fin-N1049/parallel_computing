import numpy as np
import tensorflow as tf
import pandas as pd
import zipfile
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define the path to the ZIP file
zip_path = "/content/mnist_test.csv.zip"
extract_path = "/content/"

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Identify the extracted CSV file
csv_filename = os.path.join(extract_path, "mnist_test.csv")

# Load the dataset from the extracted CSV file
data = pd.read_csv(csv_filename)

# Separate features and labels
X = data.iloc[:, 1:].values  # Pixel values
y = data.iloc[:, 0].values   # Labels

# Normalize the pixel values (0-255 -> 0-1)
X = X / 255.0

# Apply one-hot encoding to target labels
y = to_categorical(y, num_classes=10)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Reshape images to 28x28 format (for deep learning models)
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("Shapes:")
print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
print(f"Validation set: {X_val.shape}, Labels: {y_val.shape}")
print(f"Test set: {X_test.shape}, Labels: {y_test.shape}")

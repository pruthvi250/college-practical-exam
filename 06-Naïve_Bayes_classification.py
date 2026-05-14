# 6

# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Load Dataset

df = pd.read_csv("IRIS DATASET.csv")

print("Dataset Loaded Successfully")

df.head()

df.info()

# Select Features

X = df[['sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width']]

# Select Target Variable

y = df['species']

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("Training Data Shape:", X_train.shape)

print("Testing Data Shape:", X_test.shape)

# Create Gaussian Naïve Bayes Model

model = GaussianNB()

# Train Model

model.fit(X_train, y_train)

print("Model Trained Successfully")

# Predict Values

y_pred = model.predict(X_test)

print("Predicted Values:")
print(y_pred)

# Create Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

# Convert to Binary Classification

y_test_binary = (y_test == 'Iris-setosa')

y_pred_binary = (y_pred == 'Iris-setosa')

# Binary Confusion Matrix

binary_cm = confusion_matrix(
    y_test_binary,
    y_pred_binary
)

print("Binary Confusion Matrix:")
print(binary_cm)

# Extract Values

TN = binary_cm[0][0]

FP = binary_cm[0][1]

FN = binary_cm[1][0]

TP = binary_cm[1][1]

print("True Positive (TP):", TP)

print("False Positive (FP):", FP)

print("True Negative (TN):", TN)

print("False Negative (FN):", FN)

# Calculate Accuracy using Formula

accuracy_formula = (TP + TN) / (TP + TN + FP + FN)

print("Accuracy using Formula:", accuracy_formula)

# Calculate Error Rate

error_rate = 1 - accuracy

print("Error Rate:", error_rate)

# Calculate Precision using Formula

precision_formula = TP / (TP + FP)

print("Precision using Formula:", precision_formula)

# Calculate Recall using Formula

recall_formula = TP / (TP + FN)

print("Recall using Formula:", recall_formula)

# Plot Confusion Matrix

plt.figure(figsize=(6,4))

sns.heatmap(binary_cm,
            annot=True,
            fmt='d',
            cmap='Blues')

plt.xlabel("Predicted Values")

plt.ylabel("Actual Values")

plt.title("Confusion Matrix")

plt.show()
# 1

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Iris.csv")
print("Dataset Loaded Successfully")

df.head()

print("Shape of Dataset:")
print(df.shape)

print("Column Names:")
print(df.columns)

print("Missing Values in Dataset:")
print(df.isnull().sum())

df.describe()

df.info()

print("Data Types of Variables:")
print(df.dtypes)

print("Unique Species Names:")
print(df['Species'].unique())

# Create Label Encoder Object
le = LabelEncoder()
# Convert categorical values into numeric values
df['Species'] = le.fit_transform(df['Species'])
print("Categorical Variable Converted Successfully")

df.head()

# Create Scaler Object
scaler = MinMaxScaler()
# Select Numerical Columns
numeric_columns = ['SepalLengthCm',
                   'SepalWidthCm',
                   'PetalLengthCm',
                   'PetalWidthCm']

# Apply Normalization
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print("Normalization Completed Successfully")

df.head()

print(df.dtypes)
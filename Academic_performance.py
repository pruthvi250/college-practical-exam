# 2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('StudentPerformanceFactors.csv')
df.shape

df.head()

df.info()

print("Missing Values in Dataset:")
df.isnull().sum()

# Separate Numerical and Categorical Columns

numerical_cols = df.select_dtypes(include=np.number).columns

categorical_cols = df.select_dtypes(exclude=np.number).columns

# Fill Numerical Missing Values with Mean

for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill Categorical Missing Values with Mode

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing Values Handled Successfully")

df.isnull().sum()

# Check Duplicate Records

print("Duplicate Records:", df.duplicated().sum())

plt.figure(figsize=(17,5))
plt.subplot(1,2,1)
sns.distplot (df['Hours_Studied'])
plt.subplot(1,2,2)
sns.distplot(df['Attendance'])
plt.show()

df['Hours_Studied']
df.describe()

sns.boxplot(df['Hours_Studied'])

percentile25 = df['Hours_Studied'].quantile(0.25)
percentile75 = df['Hours_Studied'].quantile(0.75)

IQR = percentile75 - percentile25

lower_limit = percentile25 - 1.5 * IQR
upper_limit = percentile75 + 1.5 * IQR



print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

df[(df['Hours_Studied'] < lower_limit)]

df[(df['Hours_Studied'] > upper_limit)]

new_df = df[(df['Hours_Studied'] >= lower_limit) &
            (df['Hours_Studied'] <= upper_limit)]

new_df.shape

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
sns.distplot (df['Hours_Studied'])
plt.subplot(2,2,2)
sns.boxplot(df['Hours_Studied'])
plt.subplot(2,2,3)
sns.distplot (new_df['Hours_Studied'])
plt.subplot(2,2,4)
sns.boxplot(new_df['Hours_Studied'])
plt.show()

# Apply Log Transformation

df['Log_Hours_Studied'] = np.log(df['Hours_Studied'])

df.head()

plt.figure(figsize=(6,4))

sns.histplot(df['Hours_Studied'], kde=True)

plt.title("Before Transformation")

plt.show()

plt.figure(figsize=(6,4))

sns.histplot(df['Log_Hours_Studied'], kde=True)

plt.title("After Log Transformation")

plt.show()

df.head()
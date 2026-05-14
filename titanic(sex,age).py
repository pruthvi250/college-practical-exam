# 9

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')
titanic.head()

# Dataset Information

titanic.info()

# Missing Values
titanic.isnull().sum()

# Statistical Summary

titanic.describe()



# Survival Count Plot

plt.figure(figsize=(6,4))

sns.countplot(x='survived', data=titanic)

plt.title("Survival Count")

plt.xlabel("Survived")

plt.ylabel("Count")

plt.show()

print(titanic.columns)
print(titanic[['sex', 'age', 'survived']].head())

# Gender-wise Survival Plot

plt.figure(figsize=(6,4))

sns.countplot(x='sex',
              hue='survived',
              data=titanic)

plt.title("Survival Based on Gender")

plt.xlabel("Gender")

plt.ylabel("Count")

plt.show()

# Passenger Class Plot

plt.figure(figsize=(6,4))

sns.countplot(x='pclass', data=titanic)

plt.title("Passenger Class Distribution")

plt.xlabel("Passenger Class")

plt.ylabel("Count")

plt.show()


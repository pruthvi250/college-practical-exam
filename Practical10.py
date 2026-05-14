# 10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("IRIS DATASET.csv")
df.head()

print(df.info())

df.hist(figsize=(10,8))
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
df.boxplot()
plt.show()

for col in df.columns[:-1]:
    plt.figure()
    sns.histplot(data=df, x=col, hue='species', kde=True)
    plt.title(col)
    plt.show()

for col in df.columns[:-1]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")
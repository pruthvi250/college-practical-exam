# 8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Titanic-Dataset.csv')
df.shape

df.info()
df.isnull().sum()
df.drop(['Cabin'],axis=1,inplace=True)
df["Age"].fillna(df["Age"].mean(),inplace=True)

df["Embarked"].value_counts()
df["Embarked"].fillna("s",inplace=True)
df.isnull().sum()

df.describe()


df["Survived"].value_counts()

sns.countplot(x="Survived",data = df)
sns.countplot(x="Sex",hue="Survived",data= df )

sns.countplot(x="Pclass",hue="Survived",data= df )
sns.countplot(x="Embarked",hue="Survived",data= df )
sns.countplot(x="Age",hue="Survived",data= df )

sns.countplot(x="SibSp",hue="Survived",data= df )

sns.countplot(x="Parch",hue="Survived",data= df )
sns.countplot(x="Parch",hue="Pclass",data= df )

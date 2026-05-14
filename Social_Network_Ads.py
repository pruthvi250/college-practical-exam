# 5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Social_Network_Ads.csv')
print(dataset.head())

df = dataset.copy()

X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
print(df.isnull().sum())

print(df.cov())


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(xtrain, ytrain)

y_pred = logreg.predict(xtest)

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

cm = confusion_matrix(ytest, y_pred)

accuracy = accuracy_score(ytest, y_pred)
precision = precision_score(ytest, y_pred)
recall = recall_score(ytest, y_pred)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("TN =", TN)
print("FP =", FP)
print("FN =", FN)
print("TP =", TP)
total = TP + TN + FP + FN
print("Total =", total)

accuracy = (TP + TN) / total
print("Accuracy =", accuracy)
error_rate = (FP + FN) / total
print("Error Rate =", error_rate)
precision = TP / (TP + FP)
print("Precision =", precision)
recall = TP / (TP + FN)
print("Recall =", recall)
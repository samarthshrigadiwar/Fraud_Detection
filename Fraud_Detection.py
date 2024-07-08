#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, accuracy_score

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        exit()

data = load_data("C:/Users/Asus/Downloads/archive (5)/creditcard.csv")

print(data.head())
print(data.info())
print(data.describe())
print(data['Class'].value_counts())

# Plot class distribution
plt.figure()
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Report:\n", classification_report(y_test, y_pred_dt))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# Plot confusion matrices
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression")

plt.subplot(1, 3, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree")

plt.subplot(1, 3, 3)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest")

plt.tight_layout()
plt.show()

iso_forest = IsolationForest(contamination=0.001, random_state=42)
y_pred_iso = iso_forest.fit_predict(X_test)
y_pred_iso = np.where(y_pred_iso == 1, 0, 1)
print("Isolation Forest Accuracy:", accuracy_score(y_test, y_pred_iso))
print("Isolation Forest F1 Score:", f1_score(y_test, y_pred_iso))

one_class_svm = OneClassSVM(nu=0.001)
y_pred_svm = one_class_svm.fit_predict(X_test)
y_pred_svm = np.where(y_pred_svm == 1, 0, 1)
print("One-Class SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("One-Class SVM F1 Score:", f1_score(y_test, y_pred_svm))


# In[ ]:





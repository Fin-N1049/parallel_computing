import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exploratory Data Analysis (EDA)
sns.pairplot(pd.concat([X, y], axis=1), hue='target')
plt.show()

# Univariate Feature Selection
selector = SelectKBest(score_func=f_classif, k=2)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Feature Importance using Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

# Recursive Feature Elimination (RFE) using SVM
svm = SVC(kernel="linear")
rfe = RFE(estimator=svm, n_features_to_select=2)
rfe.fit(X_train, y_train)
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Model Evaluation
models = {
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_full = accuracy_score(y_test, y_pred)
    
    model.fit(X_train_selected, y_train)
    y_pred_selected = model.predict(X_test_selected)
    acc_selected = accuracy_score(y_test, y_pred_selected)
    
    model.fit(X_train_rfe, y_train)
    y_pred_rfe = model.predict(X_test_rfe)
    acc_rfe = accuracy_score(y_test, y_pred_rfe)
    
    print(f"{name} Accuracy: Full Features: {acc_full:.4f}, SelectKBest: {acc_selected:.4f}, RFE: {acc_rfe:.4f}")

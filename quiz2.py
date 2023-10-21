from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np

# Load MNIST data
digits = datasets.load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter tuning for SVM
param_grid_svm = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)

# Hyperparameter tuning for Decision Tree
param_grid_dt = {"criterion": ["gini", "entropy"], "max_depth": [10, 20, 30, None]}
grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5)
grid_dt.fit(X_train, y_train)

# Train models
prod_model = grid_svm.best_estimator_
cand_model = grid_dt.best_estimator_

prod_model.fit(X_train, y_train)
cand_model.fit(X_train, y_train)

# Make predictions
prod_pred = prod_model.predict(X_test)
cand_pred = cand_model.predict(X_test)

# Compute accuracies
prod_accuracy = accuracy_score(y_test, prod_pred)
cand_accuracy = accuracy_score(y_test, cand_pred)

# Compute 10x10 confusion matrix
conf_matrix_10x10 = confusion_matrix(prod_pred, cand_pred)

# Compute 2x2 confusion matrix
conf_matrix_2x2 = confusion_matrix(prod_pred == y_test, cand_pred == y_test)

# Compute macro-average F1 metrics
macro_f1_prod = f1_score(y_test, prod_pred, average="macro")
macro_f1_cand = f1_score(y_test, cand_pred, average="macro")

print("Production model's accuracy:", prod_accuracy)
print("Candidate model's accuracy:", cand_accuracy)
print("10x10 Confusion Matrix:")
print(conf_matrix_10x10)
print("2x2 Confusion Matrix:")
print(conf_matrix_2x2)
print("Macro-average F1 score for production model:", macro_f1_prod)
print("Macro-average F1 score for candidate model:", macro_f1_cand)

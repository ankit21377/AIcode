import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Load dataset
data = pd.read_csv('inflation_data.csv')

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Display column names and first few rows to verify
print("Columns in dataset:", data.columns)
print("First 5 rows:\n", data.head())

# Identify target column dynamically (contains 'inflation')
target_candidates = [col for col in data.columns if 'inflation' in col.lower()]
if len(target_candidates) == 0:
    raise ValueError("No target column containing 'inflation' found in dataset")
target_col = target_candidates[0]
print("Detected target column:", target_col)

# Map target to numeric if it's categorical
if data[target_col].dtype == 'object':
    mapping_dict = {'Low': 0, 'Medium': 1, 'High': 2}
    data[target_col] = data[target_col].map(mapping_dict)
    print("Target column after mapping:\n", data[target_col].value_counts())

# Separate features and target
X = data.drop(target_col, axis=1)
y = data[target_col]

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()
print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Normalize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split dataset into training, testing, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Validation samples:", X_val.shape[0])

# Logistic Regression with sigmoid function for multi-class classification
log_reg = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=500)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
grid = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid.best_params_)

# Train final model with best parameters
final_model = grid.best_estimator_
final_model.fit(X_train, y_train)

# Make predictions
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)

# Evaluation metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Confusion matrix heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC curves for multi-class
plt.figure(figsize=(8,6))
for i in range(len(final_model.classes_)):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


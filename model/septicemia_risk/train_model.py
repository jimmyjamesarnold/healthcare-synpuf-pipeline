import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
output_dir = os.path.join(project_root, 'outputs/model_artifacts/septicemia_risk')
os.makedirs(output_dir, exist_ok=True)

# Load dataset
dataset = pd.read_csv(os.path.join(project_root,'data/processed/septicemia_risk/dataset.csv'))

# Prepare features and target
X = dataset.drop(columns=['DESYNPUF_ID', 'index_date', 'septicemia_case_ind'])
y = dataset['septicemia_case_ind']
X = pd.get_dummies(X, columns=['BENE_SEX_IDENT_CD', 'SP_STATE_CODE'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale numeric features
scaler = StandardScaler()
numeric_cols = [col for col in X.columns if any(x in col for x in ['count', 'amt', 'age', 'chronic'])]
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Train model
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# Feature importance
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("Top 10 Features:\n", importances.head(10))
plt.figure(figsize=(10, 6))
sns.barplot(data=importances.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importances')
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# Save model and scaler
joblib.dump(model, os.path.join(output_dir,'septicemia_risk_model.pkl'))
joblib.dump(scaler, os.path.join(output_dir,'septicemia_risk_scaler.pkl'))

print("Modeling complete")
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, recall_score, f1_score
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

# Set up paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
output_dir = os.path.join(project_root, 'outputs/model_artifacts/septicemia_risk')
os.makedirs(output_dir, exist_ok=True)

# Load dataset
dataset = pd.read_csv(os.path.join(project_root, 'data/processed/septicemia_risk/dataset.csv'))

# Prepare features and target
X = dataset.drop(columns=['DESYNPUF_ID', 'index_date', 'septicemia_case_ind']).fillna(0)
y = dataset['septicemia_case_ind']
X = pd.get_dummies(X, columns=['BENE_SEX_IDENT_CD', 'SP_STATE_CODE'], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale numeric features
scaler = StandardScaler()
numeric_cols = [col for col in X.columns if any(x in col for x in ['count', 'amt', 'age', 'chronic'])]
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Apply SMOTEENN to training data
smoteenn = SMOTEENN(random_state=42)
X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train, y_train)
print(f"Original training set: {len(y_train)} samples, {sum(y_train)} positive cases")
print(f"SMOTEENN training set: {len(y_train_smoteenn)} samples, {sum(y_train_smoteenn)} positive cases")

# Define broader parameter grid for RandomizedSearchCV
param_dist = {
    'max_depth': [3, 5, 7, 10, 15],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.3, 0.5],
    'n_estimators': [100, 200, 300, 400, 500],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.5, 1.0],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# # Train model with RandomizedSearchCV (F1-score)
# model_f1 = XGBClassifier(scale_pos_weight=(len(y_train_smoteenn) - sum(y_train_smoteenn)) / sum(y_train_smoteenn), random_state=42, n_jobs=-1)
# random_search_f1 = RandomizedSearchCV(model_f1, param_distributions=param_dist, n_iter=50, scoring='f1', cv=3, random_state=42, n_jobs=-1)
# random_search_f1.fit(X_train_smoteenn, y_train_smoteenn)

# # Save GridSearchCV results
# cv_results_f1 = pd.DataFrame(random_search_f1.cv_results_)
# cv_results_f1.to_csv(os.path.join(output_dir, 'random_search_f1_results.csv'), index=False)

# Train model with RandomizedSearchCV (Recall)
model_recall = XGBClassifier(scale_pos_weight=(len(y_train_smoteenn) - sum(y_train_smoteenn)) / sum(y_train_smoteenn), random_state=42, n_jobs=-1)
random_search_recall = RandomizedSearchCV(model_recall, param_distributions=param_dist, n_iter=50, scoring='recall', cv=3, random_state=42, n_jobs=-1)
random_search_recall.fit(X_train_smoteenn, y_train_smoteenn)

# Save GridSearchCV results
cv_results_recall = pd.DataFrame(random_search_recall.cv_results_)
cv_results_recall.to_csv(os.path.join(output_dir, 'random_search_recall_results.csv'), index=False)

# Select best model (F1-score or Recall)
# Comment out one of the following blocks based on preference
# Option 1: F1-score
# model = random_search_f1.best_estimator_
# print("Best parameters (F1-score):", random_search_f1.best_params_)
# print("Best F1-score (CV):", random_search_f1.best_score_)

# Option 2: Recall (uncomment to use)
model = random_search_recall.best_estimator_
print("Best parameters (Recall):", random_search_recall.best_params_)
print("Best Recall (CV):", random_search_recall.best_score_)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Threshold tuning: Find optimal threshold for recall
thresholds = np.arange(0.05, 0.5, 0.05)
recall_scores = []
for thresh in thresholds:
    y_pred = (y_pred_proba >= thresh).astype(int)
    recall_scores.append(recall_score(y_test, y_pred))
optimal_threshold = thresholds[np.argmax(recall_scores)]
print(f"Optimal threshold for recall: {optimal_threshold:.2f}")

# Predictions with optimal threshold
y_pred = (y_pred_proba >= optimal_threshold).astype(int)
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
plt.title('ROC Curve (SMOTEENN with RandomizedSearchCV)')
plt.legend()
plt.savefig(os.path.join(output_dir, 'roc_curve_smoteenn_randomized.png'))
plt.close()

# Feature importance
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("Top 10 Features:\n", importances.head(10))
plt.figure(figsize=(10, 6))
sns.barplot(data=importances.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importances (SMOTEENN with RandomizedSearchCV)')
plt.savefig(os.path.join(output_dir, 'feature_importance_smoteenn_randomized.png'))
plt.close()

# Save model, scaler, and optimal threshold
joblib.dump(model, os.path.join(output_dir, 'septicemia_risk_model_smoteenn_randomized.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'septicemia_risk_scaler_smoteenn_randomized.pkl'))
joblib.dump(optimal_threshold, os.path.join(output_dir, 'optimal_threshold_smoteenn_randomized.pkl'))

print("Modeling with SMOTEENN and RandomizedSearchCV complete")

# Test ROC-AUC: 0.753
# Recall: 0.29, Precision: 0.05
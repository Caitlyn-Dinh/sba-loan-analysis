import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load CSV
df = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/merged_with_naics.csv', low_memory=False)

# Target column
df['Default'] = df['ChargeOffDate'].notna().astype(int)
df = df.dropna(subset=['Default'])

# Balanced sample (100,000 defaults + 100,000 non-defaults)
defaults = df[df['Default'] == 1]
non_defaults = df[df['Default'] == 0]

defaults_sample = defaults.sample(n=min(100000, len(defaults)), random_state=42)
non_defaults_sample = non_defaults.sample(n=min(100000, len(non_defaults)), random_state=42)

df = pd.concat([defaults_sample, non_defaults_sample])
print(f"Balanced dataset shape: {df.shape}")
print(df['Default'].value_counts())

# Remove leakage columns
leakage_cols = ['ChargeOffDate', 'GrossChargeOffAmount', 'PaidInFullDate', 'LoanStatus', 'FirstDisbursementDate']
df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

# Features
features = [
    'Program', 'Subprogram', 'GrossApproval', 'SBAGuaranteedApproval', 'ApprovalDate',
    'InitialInterestRate', 'FixedOrVariableInterestInd', 'TermInMonths',
    'BusinessType', 'BusinessAge', 'CollateralInd', 'RevolverStatus', 'JobsSupported', 'ProjectState'
]

X = df[features].copy()
y = df['Default']

# Process ApprovalDate
X['ApprovalDate'] = pd.to_datetime(X['ApprovalDate'], errors='coerce')
X['ApprovalYear'] = X['ApprovalDate'].dt.year
X['ApprovalMonth'] = X['ApprovalDate'].dt.month
X = X.drop(columns=['ApprovalDate'])

numeric_features = ['GrossApproval', 'SBAGuaranteedApproval', 'InitialInterestRate',
                    'TermInMonths', 'JobsSupported', 'ApprovalYear', 'ApprovalMonth']

categorical_features = [col for col in X.columns if col not in numeric_features]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) 
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# --- Metrics Storage ---
metrics_dict = {}

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def cross_val_metrics(model, model_name):
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced to 3 folds
    scoring = ['roc_auc', 'f1', 'accuracy']
    
    scores = cross_validate(clf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    metrics_dict[model_name] = {
        'AUC': scores['test_roc_auc'].mean(),
        'F1': scores['test_f1'].mean(),
        'Accuracy': scores['test_accuracy'].mean()
    }

    print(f"\n{model_name} Cross-Validation Metrics")
    print("=" * 50)
    print(f"Mean AUC: {metrics_dict[model_name]['AUC']:.4f}")
    print(f"Mean F1:  {metrics_dict[model_name]['F1']:.4f}")
    print(f"Mean Acc: {metrics_dict[model_name]['Accuracy']:.4f}")

def train_and_evaluate(model, model_name):
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print(f"\n{model_name} Test Results")
    print("=" * 50)
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, model_name)

    cross_val_metrics(model, model_name)

# Train and evaluate models
train_and_evaluate(LogisticRegression(max_iter=1000, solver='liblinear'), "Logistic Regression")
train_and_evaluate(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1), "Random Forest")
train_and_evaluate(XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1), "XGBoost")

# --- Bar Chart for Cross-Validation Metrics ---
def plot_metrics_comparison(metrics_dict):
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.plot(kind='bar', figsize=(8, 6))
    plt.title("Cross-Validation Metrics Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

plot_metrics_comparison(metrics_dict)
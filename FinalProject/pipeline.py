# Probability of Default (PD) Modeling Pipeline (Updated)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# --- Load and Sample Data ---
df = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/merged_with_naics.csv', low_memory=False)
df = df.sample(n=50000, random_state=42)

# --- Target and Leakage Removal ---
df['Default'] = df['ChargeOffDate'].notna().astype(int)
df = df.dropna(subset=['Default'])
leakage_cols = [
    'ChargeOffDate', 'GrossChargeOffAmount', 'PaidInFullDate', 'LoanStatus',
    'FirstDisbursementDate', 'AsOfDate', 'BorrName', 'LocationID', 'BankName',
    'BankFDICNumber', 'BankNCUANumber', 'BankStreet', 'BankCity', 'BankZip'
]
df = df.drop(columns=[col for col in leakage_cols if col in df.columns])

# --- Feature Selection ---
features = [
    'Program', 'Subprogram', 'GrossApproval', 'SBAGuaranteedApproval', 'ApprovalDate',
    'InitialInterestRate', 'FixedOrVariableInterestInd', 'TermInMonths',
    'BusinessType', 'BusinessAge', 'CollateralInd', 'RevolverStatus',
    'JobsSupported', 'ProjectState', 'NaicsCode', 'ProcessingMethod',
    'FranchiseCode', 'SBADistrictOffice', 'BankState'
]

X = df[features].copy()
y = df['Default']

# --- Feature Engineering ---
X['ApprovalDate'] = pd.to_datetime(X['ApprovalDate'], errors='coerce')
X['ApprovalYear'] = X['ApprovalDate'].dt.year
X['ApprovalMonth'] = X['ApprovalDate'].dt.month
X = X.drop(columns=['ApprovalDate'])

numeric_features = ['GrossApproval', 'SBAGuaranteedApproval', 'InitialInterestRate',
                    'TermInMonths', 'JobsSupported', 'ApprovalYear', 'ApprovalMonth']
categorical_features = [col for col in X.columns if col not in numeric_features]

# --- Pipelines ---
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Cross-Validation Function ---
def cross_validate_model(model, model_name):
    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    print(f"\n{model_name} Cross-Validation AUC Scores:")
    print(scores)
    print(f"Mean AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return np.mean(scores)

scores_dict = {
    "Logistic Regression": cross_validate_model(LogisticRegression(max_iter=1000), "Logistic Regression"),
    "Random Forest": cross_validate_model(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
    "XGBoost": cross_validate_model(XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42), "XGBoost")
}

# --- Visualize Cross-Validation Scores ---
plt.figure(figsize=(8, 5))
sns.barplot(x=list(scores_dict.keys()), y=list(scores_dict.values()), palette='Set2')
plt.ylim(0.85, 1.0)
plt.title("Cross-Validated AUC Comparison")
plt.ylabel("Mean AUC Score")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# --- Train Final Model and Export ---
final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42))
])
final_model.fit(X_train, y_train)

# Save predictions
test_preds = pd.DataFrame({
    'LoanID': X_test.index,
    'PredictedProb': final_model.predict_proba(X_test)[:, 1],
    'Actual': y_test.values
})
test_preds.to_csv("pd_predictions.csv", index=False)

# Save model
joblib.dump(final_model, "pd_model_xgb.joblib")

# --- ROC Curve Visualization ---
y_proba = final_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - PD Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Survival Analysis Setup ---
from lifelines import CoxPHFitter

df_surv = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/merged_with_naics.csv', low_memory=False)
df_surv = df_surv.sample(n=50000, random_state=42)
df_surv['FirstDisbursementDate'] = pd.to_datetime(df_surv['FirstDisbursementDate'], errors='coerce')
df_surv['ChargeOffDate'] = pd.to_datetime(df_surv['ChargeOffDate'], errors='coerce')
df_surv = df_surv[df_surv['FirstDisbursementDate'].notna()]
df_surv['duration'] = (df_surv['ChargeOffDate'] - df_surv['FirstDisbursementDate']).dt.days / 30

# Remove rows with negative or missing durations
df_surv = df_surv[df_surv['duration'].notna() & (df_surv['duration'] >= 0)]
df_surv['event'] = df_surv['ChargeOffDate'].notna().astype(int)

# Use same features as above
survival_features = features.copy()
survival_features.remove('ApprovalDate')  # not used
survival_features += ['duration', 'event']
df_surv = df_surv[survival_features].dropna()

# One-hot encode categorical variables
df_surv['ApprovalYear'] = pd.to_datetime(df_surv['ApprovalDate'], errors='coerce').dt.year
survival_numeric = ['GrossApproval', 'SBAGuaranteedApproval', 'InitialInterestRate', 'TermInMonths', 'JobsSupported', 'ApprovalYear']
survival_categorical = [col for col in features if col not in survival_numeric and col != 'ApprovalDate']

df_surv_encoded = pd.get_dummies(df_surv[survival_numeric + survival_categorical], drop_first=True)
df_surv_encoded['duration'] = df_surv['duration']
df_surv_encoded['event'] = df_surv['event']

# Fit Cox Model
cph = CoxPHFitter()
cph.fit(df_surv_encoded, duration_col='duration', event_col='event')
cph.print_summary()
risk_scores = cph.predict_partial_hazard(df_surv_encoded)

# --- Top Features Plot from Cox Model ---
summary_df = cph.summary
top_features = summary_df.reindex(summary_df['coef'].abs().sort_values(ascending=False).index).head(10)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_features['coef'], y=top_features.index, palette='coolwarm')
plt.axvline(0, color='black', linestyle='--')
plt.title("Top 10 Predictors of Time to Default (Cox Model)")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.show()

# --- Survival Plot ---
plt.figure(figsize=(8, 6))
cph.plot_partial_effects_on_outcome(covariates='JobsSupported', values=[0, 10, 20], cmap='coolwarm')
plt.title('Partial Effects of Jobs Supported on Survival')
plt.xlabel('Months Since Disbursement')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.tight_layout()
plt.show()

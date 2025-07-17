import pandas as pd

# Load cleaned dataset
df = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/merged_with_naics.csv', low_memory=False)

print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# === 1. Basic overview ===
print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# === 2. Missing values ===
print("\nMissing value summary:")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

# === 3. Duplicate rows ===
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates:,}")

# === 4. Top industries (NAICS) ===
print("\n Top 10 industries by number of loans:")
print(df['NaicsDescription'].value_counts().head(10))

# === 5. Top lenders ===
if 'BankName' in df.columns:
    print("\n Top 10 banks by loan count:")
    print(df['BankName'].value_counts().head(10))

# === 6. Approval amount stats ===
if 'GrossApproval' in df.columns:
    df['GrossApproval'] = pd.to_numeric(df['GrossApproval'], errors='coerce')
    print("\n Loan amount statistics (GrossApproval):")
    print(df['GrossApproval'].describe())

# === 7. Job support stats ===
if 'JobsSupported' in df.columns:
    df['JobsSupported'] = pd.to_numeric(df['JobsSupported'], errors='coerce')
    print("\n Jobs supported per loan:")
    print(df['JobsSupported'].describe())

# === 8. Charge-off analysis ===
if 'LoanStatus' in df.columns and 'GrossChargeOffAmount' in df.columns:
    df['GrossChargeOffAmount'] = pd.to_numeric(df['GrossChargeOffAmount'], errors='coerce')
    chargeoffs = df[df['LoanStatus'] == 'CHGOFF']
    print(f"\n Charge-offs: {len(chargeoffs):,} loans")
    print(" Charge-off amount statistics:")
    print(chargeoffs['GrossChargeOffAmount'].describe())

# === 9. Loans by year ===
if 'ApprovalFiscalYear' in df.columns:
    print("\nLoans by fiscal year:")
    print(df['ApprovalFiscalYear'].value_counts().sort_index())

print("\n EDA complete.")

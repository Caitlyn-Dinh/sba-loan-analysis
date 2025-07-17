import pandas as pd

# Load CSV files with low_memory=False
df1 = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/sba_project/2020-present.csv', low_memory=False)
df2 = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/sba_project/2010-2019.csv', low_memory=False)
df3 = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/sba_project/2000-2009.csv', low_memory=False)

# Rename df3 columns to match df1/df2
df3 = df3.rename(columns={
    'AsofDate': 'AsOfDate',
    'ApprovalFY': 'ApprovalFiscalYear',
    'FixedorVariableInterestRate': 'FixedOrVariableInterestInd',
    'TerminMonths': 'TermInMonths',
    'NAICSCode': 'NaicsCode',
    'NAICSDescription': 'NaicsDescription',
    'PaidinFullDate': 'PaidInFullDate',
    'ChargeoffDate': 'ChargeOffDate',
    'GrossChargeoffAmount': 'GrossChargeOffAmount',
    'SoldSecondMarketInd': 'SoldSecMrktInd'
})

# Reorder df3 columns to match df1
df3 = df3[df1.columns]

# Check column alignment
print("Column alignment checks:")
print("df1 vs df2:", df1.columns.equals(df2.columns))  # Should be True
print("df1 vs df3:", df1.columns.equals(df3.columns))  # Should be True
print()

# Merge the DataFrames
merged_df = pd.concat([df1, df2, df3], ignore_index=True)

# Save to CSV
merged_df.to_csv('/Users/caitlyndinh/ai-academy/FinalProject/merged_output.csv', index=False)

print("Files cleaned and merged into 'merged_output.csv'")

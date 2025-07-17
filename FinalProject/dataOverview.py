import pandas as pd

# Load the merged CSV file
df = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/merged_output.csv', low_memory=False)

print("DATA OVERVIEW")
print("=" * 40)
print(f" Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\n Column Names:")
print(df.columns.tolist())
print("\n Data Types:")
print(df.dtypes)
print("\n")

print("MISSING VALUE SUMMARY")
print("=" * 40)
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_summary = pd.DataFrame({
    "MissingCount": missing,
    "MissingPercent": missing_percent.map("{:.2f}%".format)
})
print(missing_summary[missing > 0].sort_values(by="MissingCount", ascending=False))
print("\n")

print("UNIQUE VALUE SUMMARY")
print("=" * 40)
for col in df.columns:
    unique_vals = df[col].nunique()
    print(f"{col}: {unique_vals} unique values")
print("\n")

print(" DESCRIPTIVE STATISTICS (Numeric Columns)")
print("=" * 40)
print(df.describe(include='number'))
print("\n")

print(" DESCRIPTIVE STATISTICS (Categorical Columns)")
print("=" * 40)
print(df.describe(include='object'))
print("\n")

print(" DUPLICATE CHECK")
print("=" * 40)
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates:,}")

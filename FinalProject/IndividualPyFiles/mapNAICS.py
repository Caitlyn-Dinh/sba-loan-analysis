import pandas as pd

# === Load both files ===
merged_df = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/merged_output.csv', low_memory=False)
naics_master = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/sba_project/naics_master.csv', dtype=str)

# === Clean and format codes as strings ===
# Remove any decimal artifacts and whitespace
merged_df['NaicsCode_clean'] = merged_df['NaicsCode'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
naics_master['NAICSCODE'] = naics_master['NAICSCODE'].astype(str).str.strip()

# === Merge on cleaned NAICS code ===
merged_df = merged_df.merge(
    naics_master[['NAICSCODE', 'NAICSTITLE']],
    how='left',
    left_on='NaicsCode_clean',
    right_on='NAICSCODE'
)

# === Apply mapped title to NaicsDescription column ===
merged_df['NaicsDescription'] = merged_df['NAICSTITLE']

# === Report rows that failed to map ===
missing_count = merged_df['NaicsDescription'].isnull().sum()
print(f"NAICS description not found for {missing_count:,} rows.")

# Optional: save failed mappings for manual review
if missing_count > 0:
    unmapped_df = merged_df[merged_df['NaicsDescription'].isnull()]
    unmapped_df.to_csv('/Users/caitlyndinh/ai-academy/FinalProject/unmapped_naics_codes.csv', index=False)
    print("Unmapped rows saved to 'unmapped_naics_codes.csv'")

# === Final cleanup ===
merged_df.drop(columns=['NAICSCODE', 'NAICSTITLE', 'NaicsCode_clean'], inplace=True)

# === Save the updated dataset ===
merged_df.to_csv('/Users/caitlyndinh/ai-academy/FinalProject/merged_with_naics.csv', index=False)
print("NAICS descriptions successfully mapped and saved to 'merged_with_naics.csv'")

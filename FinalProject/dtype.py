import pandas as pd

# Load files with low_memory=False to suppress DtypeWarnings temporarily
df1 = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/sba_project/2020-present.csv', low_memory=False)
df2 = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/sba_project/2010-2019.csv', low_memory=False)
df3 = pd.read_csv('/Users/caitlyndinh/ai-academy/FinalProject/sba_project/2000-2009.csv', low_memory=False)

# Print first few rows of each file
print("df1 (2020-present.csv):")
print(df1.head(), end="\n\n")

print("df2 (2010-2019.csv):")
print(df2.head(), end="\n\n")

print("df3 (2000-2009.csv):")
print(df3.head(), end="\n\n")

# Print column names to identify dtype mismatch sources
print("Columns in df1:")
print(df1.columns.tolist(), end="\n\n")

print("Columns in df2:")
print(df2.columns.tolist(), end="\n\n")

print("Columns in df3:")
print(df3.columns.tolist())

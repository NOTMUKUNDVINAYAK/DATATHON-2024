import pandas as pd

df = pd.read_csv('Train.csv')

missing_values_before = df.isnull().sum()
print(f"Number of missing values before preprocessing:\n{missing_values_before}")

duplicates_before = df['ENTITY_ID'].duplicated().sum()
print(f"Number of duplicates in ENTITY_ID before preprocessing: {duplicates_before}")

df.dropna(inplace=True)

missing_values_after = df.isnull().sum()
print(f"Number of missing values after preprocessing:\n{missing_values_after}")

duplicates_after = df['ENTITY_ID'].duplicated().sum()
print(f"Number of duplicates in ENTITY_ID after preprocessing: {duplicates_after}")

print(f"Number of rows after preprocessing: {df.shape[0]}")

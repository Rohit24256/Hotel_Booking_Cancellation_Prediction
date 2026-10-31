#DATA CLEANING


import pandas as pd
import numpy as np

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_csv('hotel_bookings.csv')

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())


# ==========================================
# STEP 2: DATA CLEANING
# ==========================================

# Remove duplicate rows
df_clean = df.drop_duplicates()

# Reset index
df_clean = df_clean.reset_index(drop=True)

# Handle missing values

# Fill children with 0
df_clean['children'] = df_clean['children'].fillna(0)

# Fill country with Unknown
df_clean['country'] = df_clean['country'].fillna('Unknown')

# Fill missing agent and company with 0
df_clean['agent'] = df_clean['agent'].fillna(0)
df_clean['company'] = df_clean['company'].fillna(0)


# ==========================================
# STEP 3: CHECK CLEANED DATASET
# ==========================================

print("\nOriginal dataset shape:", df.shape)
print("Cleaned dataset shape:", df_clean.shape)

print("\nRemaining duplicates:")
print(df_clean.duplicated().sum())

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())


# ==========================================
# STEP 4: SAVE CLEANED DATASET
# ==========================================

df_clean.to_csv('hotel_bookings_cleaned.csv', index=False)

print("\nCleaned dataset saved successfully!")

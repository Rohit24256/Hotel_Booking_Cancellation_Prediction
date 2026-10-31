#FEATURE SELECTION



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier


# ==========================================
# STEP 3.1: LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv('hotel_bookings_cleaned.csv')

print("Dataset shape:")
print(df.shape)


# ==========================================
# STEP 3.2: CORRELATION ANALYSIS
# ==========================================

df_numeric = df.select_dtypes(include=['number'])

correlation = df_numeric.corr()

print("\nCorrelation with Cancellation:")
print(correlation['is_canceled'].sort_values(ascending=False))


# ==========================================
# STEP 3.3: CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(14, 9))

sns.heatmap(
    correlation,
    cmap='rocket',
    annot=True,
    fmt='.2f'
)

plt.title('Correlation Heatmap')

plt.show()


# ==========================================
# STEP 3.4: SEPARATE FEATURES AND TARGET
# ==========================================

# Remove target and leakage columns
X = df.drop(
    ['is_canceled', 'reservation_status',
     'reservation_status_date', 'agent', 'company'],
    axis=1
)

y = df['is_canceled']

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True)

print("\nFeatures shape:")
print(X.shape)


# ==========================================
# STEP 3.5: RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X, y)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)


# ==========================================
# STEP 3.6: SELECT TOP 10 FEATURES
# ==========================================

top_features = importance.head(10)

print("\nTop 10 Selected Features:")
print(top_features)


# ==========================================
# STEP 3.7: FEATURE IMPORTANCE GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_features,
    x='Importance',
    y='Feature',
    hue='Feature',
    palette='cividis',
    legend=False
)

plt.title('Top 10 Important Features')
plt.xlabel('Importance Score')
plt.ylabel('Features')

plt.show()


# ==========================================
# STEP 3.8: SAVE SELECTED FEATURES
# ==========================================

selected_features = top_features['Feature'].tolist()

X_selected = X[selected_features].copy()

X_selected['is_canceled'] = y.values

X_selected.to_csv(
    'selected_features_dataset.csv',
    index=False
)

print("\nSelected features saved successfully!")

print("\nFinal selected dataset shape:")
print(X_selected.shape)

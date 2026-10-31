#MODEL TRAINING 



import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score


# ==========================================
# STEP 4.1: LOAD SELECTED FEATURES DATASET
# ==========================================

df = pd.read_csv('selected_features_dataset.csv')

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# ==========================================
# STEP 4.2: SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

print("\nFeatures:")
print(X.columns)

print("\nTarget variable:")
print(y.head())


# ==========================================
# STEP 4.3: SPLIT TRAINING AND TESTING DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# STEP 4.4: FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# STEP 4.5: LOGISTIC REGRESSION
# ==========================================

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_scaled, y_train)

lr_predictions = lr_model.predict(X_test_scaled)

print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, lr_predictions))


# ==========================================
# STEP 4.6: DECISION TREE
# ==========================================

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

print("\nDecision Tree Accuracy:",
      accuracy_score(y_test, dt_predictions))


# ==========================================
# STEP 4.7: RANDOM FOREST
# ==========================================

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:",
      accuracy_score(y_test, rf_predictions))


# ==========================================
# STEP 4.8: K-NEAREST NEIGHBORS
# ==========================================

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

knn_predictions = knn_model.predict(X_test_scaled)

print("\nKNN Accuracy:",
      accuracy_score(y_test, knn_predictions))


# ==========================================
# STEP 4.9: COMPLETION
# ==========================================

print("\nAll four models trained successfully!")

#MODEL COMPARISON



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# ==========================================
# STEP 5.1: LOAD DATASET
# ==========================================

df = pd.read_csv('selected_features_dataset.csv')

X = df.drop('is_canceled', axis=1)
y = df['is_canceled']


# ==========================================
# STEP 5.2: SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# STEP 5.3: FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# STEP 5.4: TRAIN FOUR ML MODELS
# ==========================================

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = []

for name, model in models.items():

    if name in ['Logistic Regression', 'KNN']:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append([
        name, accuracy, precision, recall, f1
    ])

    print("\nModel:", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


# ==========================================
# STEP 5.5: MODEL COMPARISON TABLE
# ==========================================

comparison = pd.DataFrame(
    results,
    columns=[
        'Model',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score'
    ]
)

print("\nMODEL COMPARISON:")
print(comparison)


# ==========================================
# STEP 5.6: MODEL COMPARISON GRAPH
# ==========================================

comparison.set_index('Model').plot(
    kind='bar',
    figsize=(12, 6),
    color=['#A78BFA','#F472B6','#FB923C','#34D399'],
    edgecolor='black',linewidth=1
)

plt.title('Performance Comparison of ML Models')
plt.xlabel('Machine Learning Models')
plt.ylabel('Performance Score')

plt.xticks(rotation=20)
plt.legend(loc='lower right')
plt.tight_layout()

plt.show()


# ==========================================
# STEP 5.7: CONFUSION MATRICES
# ==========================================

for name, model in models.items():

    if name in ['Logistic Regression', 'KNN']:
        predictions = model.predict(X_test_scaled)

    else:
        predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='icefire'
    )

    plt.title(name + ' Confusion Matrix')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')

    plt.show()


# ==========================================
# STEP 5.8: IDENTIFY BEST MODEL
# ==========================================

best_model = comparison.loc[
    comparison['F1 Score'].idxmax()
]

print("\nBEST PERFORMING MODEL:")
print(best_model)

print("\nModel comparison completed successfully!")

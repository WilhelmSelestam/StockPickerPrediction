import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load the dataset
df = pd.read_csv('dataset.csv')  # Adjust the path if needed
print("Dataset shape:", df.shape)
print(df.head())

# 2. Data Cleaning
# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop non-feature columns (e.g., date, ticker, and any future-lookahead columns)
cols_to_drop = ['date', 'ticker', 'INCREMENTO', 'diff']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# Impute missing numeric values with median (this handles NaNs from early windows)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Verify that there are no missing values left
print("Total missing values after cleaning:", df.isnull().sum().sum())

# 3. Prepare features and target variable
# Assume the target column is named 'TARGET'
X = df.drop('TARGET', axis=1)
y = df['TARGET']

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# 4. Split the data into training and testing sets
# Use stratification to preserve the class distribution (e.g., around 18% TARGET=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train a Random Forest classifier
rf = RandomForestClassifier(
    n_estimators=100,        # Number of trees in the forest
    random_state=42,
    class_weight='balanced'  # Helps deal with class imbalance
)
rf.fit(X_train, y_train)

# Evaluate on the test set
y_pred = rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Plot a confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, [0, 1])
plt.yticks(tick_marks, [0, 1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# Annotate the confusion matrix
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# 7. Plot feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(12, 6))
plt.title('Feature Importances from Random Forest')
plt.bar(range(len(feature_names)), importances[indices], align='center')
plt.xticks(range(len(feature_names)), feature_names[indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

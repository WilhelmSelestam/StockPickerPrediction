import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import shap

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

# 4. Split the data into training and testing sets (stratify to preserve class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train a Random Forest classifier on original data
rf = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    random_state=42,
    class_weight='balanced'  # Helps with class imbalance
)
rf.fit(X_train, y_train)

# Evaluate on the test set with default threshold (0.5)
y_pred = rf.predict(X_test)
print("\nClassification Report (Default Threshold):")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall (Default) for jump events:", recall_score(y_test, y_pred))

# 6. Adjust threshold to improve recall
# Get predicted probabilities for the positive class (jump events)
y_proba = rf.predict_proba(X_test)[:, 1]
threshold = 0.3  # Lower the threshold from 0.5 to catch more jump events
y_pred_adjusted = (y_proba >= threshold).astype(int)
print("\nClassification Report (Adjusted Threshold = {}):".format(threshold))
print(classification_report(y_test, y_pred_adjusted))
print("Recall (Adjusted) for jump events:", recall_score(y_test, y_pred_adjusted))

# Optionally, plot Precision-Recall curve to help choose a threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall for Different Thresholds')
plt.legend()
plt.grid(True)
plt.show()

# 7. Apply SMOTE to improve recall by oversampling the minority class
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("After SMOTE, X_train shape:", X_train_sm.shape, "y_train shape:", y_train_sm.shape)

rf_sm = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
rf_sm.fit(X_train_sm, y_train_sm)
y_pred_sm = rf_sm.predict(X_test)
print("\nClassification Report (After SMOTE):")
print(classification_report(y_test, y_pred_sm))
print("Recall (SMOTE) for jump events:", recall_score(y_test, y_pred_sm))

# Plot confusion matrix for the SMOTE model
cm_sm = confusion_matrix(y_test, y_pred_sm)
plt.figure(figsize=(6,4))
plt.imshow(cm_sm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (SMOTE)')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, [0, 1])
plt.yticks(tick_marks, [0, 1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
thresh = cm_sm.max() / 2.
for i in range(cm_sm.shape[0]):
    for j in range(cm_sm.shape[1]):
        plt.text(j, i, format(cm_sm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_sm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# 8. Explainability using SHAP
# Create an explainer for the original Random Forest model
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Plot a summary bar chart for the jump class (class index 1)
plt.figure()
shap.summary_plot(shap_values[1], X_test, plot_type="bar")
plt.show()

# Code Improvements Guide

This document outlines recommended code improvements for the Triage classification project.

## 🔧 Priority Fixes

### 1. Fix Pandas FutureWarnings

**Issue:** The code uses deprecated `inplace=True` with chained assignment, which will break in pandas 3.0.

**Current problematic code:**
```python
df['Chief_complain'].fillna('', inplace=True)
df[col].fillna(df[col].median(), inplace=True)
```

**Recommended fix:**
```python
# Instead of inplace operations on series, modify the dataframe directly
df = df.copy()  # Ensure we're working with a copy
df['Chief_complain'] = df['Chief_complain'].fillna('')

# For numerical columns
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())
```

### 2. Replace Hardcoded File Paths

**Issue:** The code uses Windows-specific hardcoded paths.

**Current code:**
```python
df = pd.read_csv(r"C:\DATASETS\triage\data.csv", delimiter=';', encoding='windows-1254')
```

**Recommended fix:**
```python
import os
from pathlib import Path

# Use relative path
data_path = Path('data.csv')
# Or use os module for cross-platform compatibility
# data_path = os.path.join(os.path.dirname(__file__), 'data.csv')

if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at {data_path}. "
        "Please ensure 'data.csv' is in the project root directory."
    )

df = pd.read_csv(data_path, delimiter=';', encoding='windows-1254')
```

## 📝 Code Quality Improvements

### 3. Add Cell Documentation

**Add markdown cells to explain each major section:**

```markdown
# Data Loading and Preprocessing
This section loads the medical triage dataset and performs initial cleaning.

# Feature Engineering
We create features from both numerical vitals and text-based chief complaints.

# Model Training
Training the Random Forest classifier with SMOTE for class imbalance.

# Model Evaluation
Assessing model performance on the test set.
```

### 4. Error Handling

**Add proper error handling:**

```python
try:
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Error: 'imbalanced-learn' library is required.")
    print("Install it using: pip install imbalanced-learn")
    raise

try:
    df = pd.read_csv(data_path, delimiter=';', encoding='windows-1254')
except FileNotFoundError:
    print(f"Error: Dataset file not found at {data_path}")
    print("Please place 'data.csv' in the project root directory.")
    raise
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise
```

### 5. Configuration Variables

**Move magic numbers and configurations to the top:**

```python
# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 1

# Use these variables throughout
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

final_model_with_smote = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('classifier', RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
        n_estimators=N_ESTIMATORS
    ))
])
```

## 🚀 Performance Improvements

### 6. Model Optimization

**Consider hyperparameter tuning:**

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    final_model_with_smote,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

### 7. Feature Importance Analysis

**Add feature importance visualization:**

```python
import matplotlib.pyplot as plt

# Get feature importance
feature_importance = final_model_with_smote.named_steps['classifier'].feature_importances_

# Get feature names (requires some preprocessing)
feature_names = numeric_features + ['chief_complaint_feature_' + str(i) 
                                    for i in range(len(feature_importance) - len(numeric_features))]

# Plot top 20 features
top_indices = feature_importance.argsort()[-20:][::-1]
plt.figure(figsize=(10, 8))
plt.barh(range(20), feature_importance[top_indices])
plt.yticks(range(20), [feature_names[i] for i in top_indices])
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features for Triage Classification')
plt.tight_layout()
plt.show()
```

## 📊 Additional Evaluation Metrics

### 8. Confusion Matrix Visualization

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate predictions
y_pred = final_model_with_smote.predict(X_test)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for KTAS Level Prediction')
plt.show()
```

### 9. Cross-Validation Scores

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(
    final_model_with_smote, X, y, cv=5, scoring='accuracy'
)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

## 🔍 Data Quality Checks

### 10. Add Data Validation

```python
# Check for data quality issues
print("Data Quality Report:")
print(f"Total records: {len(df)}")
print(f"\nMissing values by column:")
print(df.isnull().sum())
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nClass distribution:")
print(df['KTAS_expert'].value_counts().sort_index())
print(f"\nData types:")
print(df.dtypes)
```

## 💾 Model Persistence

### 11. Save and Load Model

```python
import joblib
from datetime import datetime

# Save the trained model
model_filename = f'triage_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
joblib.dump(final_model_with_smote, model_filename)
print(f"Model saved to {model_filename}")

# Load the model
# loaded_model = joblib.load(model_filename)
# predictions = loaded_model.predict(X_test)
```

## 📝 Summary

Implementing these improvements will:
- ✅ Fix deprecation warnings
- ✅ Make code portable across different systems
- ✅ Improve code readability and maintainability
- ✅ Add proper error handling
- ✅ Enable better model evaluation
- ✅ Allow for model persistence and reuse

## 🔄 Next Steps

1. **Immediate**: Fix pandas warnings and hardcoded paths
2. **Short-term**: Add documentation and error handling
3. **Medium-term**: Implement hyperparameter tuning
4. **Long-term**: Consider deep learning approaches for comparison

---

**Note**: These improvements should be implemented locally in Jupyter Notebook. Test each change incrementally to ensure the model performance remains consistent.

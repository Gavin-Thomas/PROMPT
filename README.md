
### Variables that could help delineate our cognitive categories further
-  physician_main (physician name)
-  patient_taking (aricept, galantamine, rivastigmine, memantine, or none)
-  cognitive_cat (we have this variable, but it is empty -- this was a text variable -- this is why I made cognitive categories using Dr. Smith's rules
-  

  
-  smoking_current_duration
       
# Final Combined Normalized IADL Scores

## Overview
I believe I have now fixed most of the issues that we were having with the IADL scores, as it was a problem in my calculations that led to the odd graphs we were seeing. I have now fixed this. Please see below for the report.

### Steps I Took:
1. **Normalization of New IADL Scores**:
   - Raw scores (0–23) were normalized using: raw score / 23 *100
   - This scales the New IADL scores to a percentage (0–100%), where 100% reflects perfect function.

2. **Normalization of Old IADL Scores**:
   - Raw scores (7–21) were normalized using: raw score / (21-7) * 100
   - Here, 7 (perfect score) maps to **100%**, and 21 (worst score) maps to **0%**, reversing the original scale.

3. **Combining Scores**:
   - New IADL scores were prioritized when available.
   - If New IADL scores were missing, Old IADL scores were used.

4. **Visualization**:
   - A histogram with kernel density estimation (KDE) was created to display the distribution of the combined normalized scores across cognitive categories.
![Combined Normalized IADL Scores](https://github.com/Gavin-Thomas/PROMPT/blob/main/images/IADL.png?raw=true)
**Figure 1. Distribution of Combined Normalized IADL Scores**  
This histogram shows the distribution of combined normalized IADL scores across cognitive categories (Definite Normal, MCI, Dementia). Scores are normalized to a 0–100% scale, where higher values indicate better function. New IADL scores were prioritized when available, with Old IADL scores used otherwise. Definite Normal individuals cluster near 100%, MCI individuals are spread across 50–80%, and Dementia individuals are concentrated in the lower range (0–40%).

## Interpretation of the Histogram
The histogram of combined normalized IADL scores shows:
- **Definite Normal** individuals cluster near **100%**, reflecting high functional independence.
- **Definite MCI** individuals are broadly distributed, with most scores between **50–80%**, consistent with moderate functional ability.
- **Definite Dementia** individuals cluster at the lower end (0–40%), indicating severe impairment.

## Summary Table

| Metric                  | Old IADL         | New IADL         | Combined Normalized IADL |
|-------------------------|------------------|------------------|--------------------------|
| **Count**              | 681              | 649              | 1329                     |
| **Mean**               | 9.85             | 17.15            | 77.16                    |
| **Standard Deviation** | 3.31             | 6.07             | 25.13                    |
| **Minimum**            | 7                | 0                | 0.00                     |
| **25th Percentile**    | 7                | 14               | 64.29                    |
| **Median**             | 9                | 19               | 85.71                    |
| **75th Percentile**    | 12               | 23               | 100.00                   |
| **Maximum**            | 21               | 23               | 100.00                   |


## IADL Women Score Distribution Relative to Men

### Graphs
![Combined Normalized IADL Scores MEN](https://github.com/Gavin-Thomas/PROMPT/blob/main/images/IADL%20MEN.png?raw=true)
   This graph shows the density of normalized functional scores for men, grouped by definite cognitive categories.

![Combined Normalized IADL Scores WOMEN](https://github.com/Gavin-Thomas/PROMPT/blob/main/images/IADL%20WOMEN.png?raw=true) 
   This graph highlights the distribution of normalized functional scores for women across definite cognitive categories.

#### Men
- **Definite Normal**:
  - Scores peak near **100%**, indicating high functional independence.
- **Definite MCI**:
  - Scores are broadly distributed between **50–80%**, reflecting moderate independence.
- **Definite Dementia**:
  - Scores cluster in the lower range (**0–40%**), signifying severe impairment.

#### Women
- **Definite Normal**:
  - Similarly concentrated near **100%**, reflecting high functional ability.
- **Definite MCI**:
  - Distributed across the mid-range, showing consistent patterns with men.
- **Definite Dementia**:
  - Scores align with men, clustering in the **0–40%** range.
 
WITH THE SCORING, I NOTICED THAT BOTH MEN AND WOMEN WERE SCORED THE SAME WAY. SO THERE IS NO DISCREPANCY IN SCORING.

# MoCA and MMSE Total Scores by Cognitive Categories

## Overview
This section examines the distributions of **MoCA Total Scores** and **MMSE Total Scores** across cognitive categories (Definite Normal, Definite MCI, and Definite Dementia). These assessments provide critical insights into cognitive performance, with higher scores reflecting better cognition.

### MoCA Total Scores by Cognitive Categories
**Description**: The distribution of MoCA scores highlights distinct patterns across cognitive categories. Definite Normal individuals cluster at the upper range, while Definite Dementia scores are concentrated at the lower end.
![MoCA Total Scores](https://github.com/Gavin-Thomas/PROMPT/blob/main/images/MOCA.png?raw=true)

#### **MoCA Total Scores**:
- **Definite Normal**:
  - Scores peak in the range of **27–30**, indicating intact cognitive function.
- **Definite MCI**:
  - Broadly distributed across the mid-range (**20–25**), consistent with mild impairments.
- **Definite Dementia**:
  - Concentrated in the lower range (**0–15**), reflecting significant cognitive decline.

### MMSE Total Scores by Cognitive Categories
**Description**: The MMSE scores show similar patterns to the MoCA, with clear differentiation between cognitive categories. Definite Normal scores peak near 30, while Definite Dementia clusters at lower scores.
![MMSE Total Scores](https://github.com/Gavin-Thomas/PROMPT/blob/main/images/MMSE.png?raw=true)

#### **MMSE Total Scores**:
- **Definite Normal**:
  - Peaks near **28–30**, aligning with preserved cognition.
- **Definite MCI**:
  - Scores mostly range between **24–27**, showing mild impairments.
- **Definite Dementia**:
  - Scores are heavily clustered in the **0–20** range, indicating severe cognitive impairment.

### MMSE and MoCA Scores by Sex

![MMSE Total Scores](https://github.com/Gavin-Thomas/PROMPT/blob/main/images/MMSEMOCA-SUBPLOTS.png?raw=true)
MoCA scores for men and women show similar trends, with distinct peaks for Definite Normal near 30 and broad distributions for Definite MCI. Minor gender differences are observed in variability for Definite Dementia. MMSE scores align closely for men and women, with both groups showing clear separation between cognitive categories. Minor differences in variability are observed for Definite MCI and Definite Dementia.

---
# Classification of Binary Dementia Categores

- I made a category called dementia_binary
- If the individual had possible or definite dementia, they received a '1', if it was anything else, they received a zero
- I trained a binary logistic regression model, with cross validation, on the PROMPT dataset
  
### Below is my code for the model

**_Quick summary:_**
I processed the dataset by removing irrelevant columns, such as identifiers and consent-related information, to avoid bias and data leakage. I split the features into numerical and categorical types. Numerical data was imputed using the median and standardized for better model performance, while categorical data was imputed with the most frequent values and one-hot encoded. To refine the feature set, I used a Random Forest classifier to identify important features, even though the final model was a logistic regression. I split the data into training and testing sets, ensuring class distribution was maintained. To address class imbalance, I applied SMOTE to oversample the minority class (the dementia class). I then used GridSearchCV to tune hyperparameters, focusing on the regularization parameter \( C \) and penalty type. The best-performing model used a \( C \) value of 0.00077 and an L2 penalty, favoring strong regularization to prevent overfitting. The model achieved a sensitivity of 75.34% and a specificity of 78.24%, showing a decently balanced ability to identify both dementia and non-dementia cases. The ROC AUC score was 0.8205. 

Go past this code to see the graphical results.

~~~
# logistic_regression_dementia.py

# Import necessary libraries
import pandas as pd
import numpy as np

# Preprocessing and modeling
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Model saving
import joblib

# Handle imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def main():
    # 1. Load the dataset
    try:
        data = pd.read_csv('fully_cleaned_combined_normalized_iadl_no_nan.csv')  
        print("Dataset loaded successfully.\n")
        print("Columns in the dataset:")
        print(data.columns.tolist())
    except FileNotFoundError:
        print("Error: Dataset file not found in the current directory.")
        return

    # 2. Define target and predictors
    target = 'dementia_binary'

    # Columns to exclude from predictors
    excluded_columns = [
        'prompt_id',
        'prompt_consent_status',
        'date_of_consent_complete',
        'informant_relation',
        'live_with_informant',
        'handedness'
    ]

    # 3. Define initial predictor columns
    predictors = [col for col in data.columns if col not in excluded_columns + [target]]
    print(f"\nNumber of predictors before excluding 'dx_imp' columns: {len(predictors)}")

    # Exclude any predictors that start with 'dx_imp'
    predictors = [col for col in predictors if not col.startswith('dx_imp')]
    print(f"Number of predictors after excluding 'dx_imp' columns: {len(predictors)}")

    # Ensure 'dementia_binary' is not in predictors
    if target in predictors:
        print(f"Error: Target variable '{target}' is still in the predictors list!")
        return

    # Exclude any columns containing 'dementia' in their name
    suspect_columns = [col for col in predictors if 'dementia' in col.lower()]
    if suspect_columns:
        print(f"\nWarning: Found columns related to the target variable: {suspect_columns}")
        predictors = [col for col in predictors if col not in suspect_columns]
        print(f"Number of predictors after removing suspect columns: {len(predictors)}")

    # 4. Set up features (X) and target (y)
    X = data[predictors]
    y = data[target]

    # 5. Specify numerical columns (modifiable for flexibility)
    numerical_cols = [
        'education_years',
        'smoking_former_agestart',
        'smoking_former_agestop',
        'smoking_current_qty',
        'rudas_vsoltotalscore',
        'mmsetotal',
        'moca_total',
        'iadltotal_old',
        'iadltotal_new',
        'Combined_Normalized_IADL',
        'age'
    ]

    # Ensure specified numerical columns exist in the data
    existing_numerical = [col for col in numerical_cols if col in X.columns]
    missing_numerical = [col for col in numerical_cols if col not in X.columns]
    if missing_numerical:
        print(f"\nWarning: Missing numerical columns will be excluded: {missing_numerical}")

    # Define categorical columns
    categorical_cols = [col for col in predictors if col not in existing_numerical]
    print(f"Number of numerical predictors: {len(existing_numerical)}")
    print(f"Number of categorical predictors: {len(categorical_cols)}")

    # 6. Identify and exclude highly correlated numerical features
    correlation_threshold = 0.99  # Adjust as needed

    # Compute correlation matrix
    correlation_matrix = data[existing_numerical + [target]].corr()
    target_correlation = correlation_matrix[target].drop(target).abs()
    high_corr_features = target_correlation[target_correlation >= correlation_threshold].index.tolist()

    if high_corr_features:
        print(f"\nHighly correlated numerical features with '{target}': {high_corr_features}")
        for feature in high_corr_features:
            existing_numerical.remove(feature)
            print(f"Excluded '{feature}' due to high correlation with target.")

    # 7. Identify and exclude categorical features that perfectly predict the target
    problematic_categorical_features = []
    for col in categorical_cols:
        cross_tab = pd.crosstab(data[col], y)
        for category in cross_tab.index:
            if (cross_tab.loc[category] == 0).any():
                problematic_categorical_features.append(col)
                print(f"Excluded '{col}' due to perfect prediction in category '{category}'.")
                break  # Only need to find one category that perfectly predicts

    # Remove problematic categorical features
    categorical_cols = [col for col in categorical_cols if col not in problematic_categorical_features]
    print(f"Number of categorical predictors after exclusion: {len(categorical_cols)}")

    # 8. Check for features identical to the target
    for col in existing_numerical + categorical_cols:
        if data[col].equals(y):
            print(f"Feature '{col}' is identical to the target. Excluding it.")
            if col in existing_numerical:
                existing_numerical.remove(col)
            else:
                categorical_cols.remove(col)

    # 9. Handle missing values and encode categorical variables
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, existing_numerical),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # 10. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print("\nData split into training and testing sets.")

    # 11. Create a function to build pipelines for different models
    def create_pipeline(model):
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
            ('classifier', model)
        ])
        return pipeline

    # 12. Define models and parameter grids
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(
                solver='saga', max_iter=10000, random_state=42),
            'param_grid': {
                'classifier__C': np.logspace(-4, 4, 10),
                'classifier__penalty': ['l1', 'l2']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__class_weight': ['balanced']
            }
        },
        'XGBoost': {
            'model': XGBClassifier(
                use_label_encoder=False, eval_metric='logloss', random_state=42),
            'param_grid': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7],
                'classifier__scale_pos_weight': [1, 2, 5]  # Adjust for imbalance
            }
        }
    }

    # 13. Cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 14. Fit models with hyperparameter tuning and evaluate
    best_models = {}
    for model_name, model_info in models.items():
        print(f"\nTraining and tuning {model_name}...")
        pipeline = create_pipeline(model_info['model'])
        param_grid = model_info['param_grid']

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest Parameters for {model_name}:")
        print(grid_search.best_params_)
        print(f"\nBest Cross-Validation ROC AUC Score for {model_name}: {grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model

        # Evaluate on test set
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        # Adjust classification threshold
        threshold = 0.5  # Adjust as needed
        y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred_adjusted)
        tn, fp, fn, tp = cm.ravel()

        # Calculate sensitivity and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"\nModel Evaluation for {model_name} at Threshold {threshold}:")
        print("\nConfusion Matrix:")
        print(cm)
        print(f"\nSensitivity (Recall for positive class): {sensitivity:.4f}")
        print(f"Specificity (Recall for negative class): {specificity:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_adjusted, digits=4))

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC Score: {roc_auc:.4f}")

        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 15. Adjust Threshold to Improve Sensitivity and Specificity
        thresholds = np.arange(0.0, 1.0, 0.01)
        sensitivities = []
        specificities = []

        for thresh in thresholds:
            y_pred_thresh = (y_pred_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivities.append(sensitivity)
            specificities.append(specificity)

        # Plot Sensitivity vs. Specificity
        plt.figure(figsize=(8,6))
        plt.plot(thresholds, sensitivities, label='Sensitivity')
        plt.plot(thresholds, specificities, label='Specificity')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Sensitivity and Specificity at Different Thresholds - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Feature Importance (for models that support it)
        if model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
            # Get feature names
            preprocessor = best_model.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()

            # Get selected features
            mask = best_model.named_steps['feature_selection'].get_support()
            selected_features = feature_names[mask]

            if model_name == 'Logistic Regression':
                coefficients = best_model.named_steps['classifier'].coef_[0]
                importance = np.abs(coefficients)
            else:
                importance = best_model.named_steps['classifier'].feature_importances_

            # Create a DataFrame for visualization
            coef_df = pd.DataFrame({
                'Feature': selected_features,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)

            # Plot top 20 features
            plt.figure(figsize=(10, 8))
            sns.barplot(
                x='Importance', y='Feature', data=coef_df.head(20),
                palette='viridis', orient='h')
            plt.title(f'Top 20 Features - {model_name}')
            plt.tight_layout()
            plt.show()

    # 16. Save the best models
    for model_name, model in best_models.items():
        model_filename = f'{model_name.lower().replace(" ", "_")}_dementia_model.joblib'
        joblib.dump(model, model_filename)
        print(f"\nModel saved as '{model_filename}'.")

if __name__ == "__main__":
    main()
~~~

## Best Logistic Regression Parameters
The best parameters for the logistic regression model were determined as:  
`{'classifier__C': 0.000774263682681127, 'classifier__penalty': 'l2'}`  

The best cross-validation ROC AUC score achieved was **0.8638**.

---

## Model Evaluation for Logistic Regression at Threshold 0.5

### AUC-ROC Curve
<div style="text-align: left;">
  <img src="https://github.com/Gavin-Thomas/PROMPT/blob/main/images/LR-BINARY.png?raw=true" alt="AUC-ROC Curve" style="width: 800px;">
</div>

---

### Confusion Matrix
<div style="text-align: left;">
  <img src="https://github.com/Gavin-Thomas/PROMPT/blob/main/images/CM-LR.png?raw=true" alt="Confusion Matrix" style="width: 800px;">
</div>

- **Sensitivity (Recall for positive class): 0.7534**  
- **Specificity (Recall for negative class): 0.7824**

---

### Sensitivity and Specificity by Threshold
<div style="text-align: left;">
  <img src="https://github.com/Gavin-Thomas/PROMPT/blob/main/images/Threshold-Sens-Spec.png.png?raw=true" alt="Sensitivity and Specificity by Threshold" style="width: 800px;">
</div>

---

### ROC AUC Score
The ROC AUC score for this model is **0.8205**.

---

### Classification Report

| Class          | Precision | Recall  | F1-Score | Support |
|----------------|-----------|---------|----------|---------|
| No Dementia (0)| 0.8935    | 0.7824  | 0.8343   | 193     |
| Dementia (1)   | 0.5670    | 0.7534  | 0.6471   | 73      |
| **Accuracy**   |           |         | 0.7744   | 266     |
| **Macro Avg**  | 0.7303    | 0.7679  | 0.7407   | 266     |
| **Weighted Avg**| 0.8039   | 0.7744  | 0.7829   | 266     |

---

## Feature Importance
<div style="text-align: left;">
  <img src="https://github.com/Gavin-Thomas/PROMPT/blob/main/images/LR-importance.png?raw=true" alt="Feature Importance" style="width: 800px;">
</div>

### In Order of Importance (Most to Least)

1. **The overall IADL score** - normalized  
2. **MMSE total score**  
3. **MOCA total score**  
4. **NEW IADL total score**  
5. **OLD IADL total score**  
6. **Age**  
7. **A recall score of 0** on the MOCA recall subtest (0 indicates failure on all recalls)  
8. **Orientation/Location test**: "Where are we?" (Country, province, city, name of hospital or street, floor #)  
   - Maximum points: 5 (great performance)  
9. **Recalls all 3 objects** given in the MOCA subtest (great performance)  
10. **Max points** on the MOCA orientation subtask  
11. **Max points** on the IADL transport task in the NEW IADL (good performance for transport)  
12. **MMSE "What is the date?" task**: Year, season, month, date, day  
    - Maximum score: 5 (max performance)  
13. **Full function in the shopping category** on the NEW IADL subscale  
14. **Full function in all food-related tasks** - NEW IADL subscale (food)  
15. **Full points in managing money** on the NEW IADL (good score)  
16. **Participant fails the MMSE copy test**:  
    - "Show the patient a diagram of intersecting pentagons and ask them to copy it. Score 1 point if all 10 angles are present and the overlap of pentagons includes four sides."  
17. **Participant passes the MMSE copy test** (see above)  
18. **Full responsibility for managing own medications** - NEW IADL  

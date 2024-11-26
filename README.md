# PROMPT Preliminary Model Analysis

## Overview

The following outlines my initial analysis of the data using just the PROMPT Registry. I had 4 outcome categories:
1. Normal Cognition
2. MCI
3. Dementia
4. Other

The category **_Other_** Included outcomes that were NOT possible or definite MCI, Dementia, or Cognitively Normal. They could have been combinations of the above (i.e. possible normal or psych) or they could have been different altogether (possible psych).

My process involved 3 models. I started with multinomial logistic regression, then moved on to a random forest model, then finally moved to XGBoost (which is a GBM). This was to follow increasing complexity. For the sake of this preliminary analysis, I DID NOT use any Ensemble ML stacking techniques to find the best meta-model given a set of base models.

For predictor variables, I did not use any variables that had the string format for values (word description variables). I also did not include any tempora variables (date of visit, etc.). Finally, I did NOT include any variables that hinted at diagnosis made by physicians (the dx_imp variables - which I used to make my 4 cognitive categories above based off Dr. Smith's Rules). 

For my target variable, I used overall_diag_main -- this variable I created... and it was determined from the Cognitive_Category outcomes (i.e. Possible MCI, Definite Dementia, etc.). For this variable I did not distinguish between possible and definite diagnoses, I also did not distinguish between severity of dementia. Essentially it's just a simpler version for our models

**BEFORE I START, BIG CAVEAT....**
- This version of these models is the absolute worst they will be, as I was quite stringent to not 'cheat' with the types of variables I gave the models
- In the Jaakamainen paper, the variables they used were HIGHLY related to diagnosis of Alzheimer's disease and related dementias. For example, their best performing model included dementia medications like memantine, and diagnostic codes in DAD related to dementia -- if I also include these when we incorporate the administrative health records, I have zero doubt in my mind that our models will perform better than the Jaakamainen model.
- Also, if you see low precision and recall relative to Jaakamainen, it is because of a few LARGE factors. The first being that we have more categories (4 Categories, instead of Jaakamainens 2). Therefore Random chance would be an accuracy of 25%, so anything above that is better than random chance. Whereas for the Jaakamainen model, random chance woulld be 50%. Also, as I said we did not include related diagnosis codes. If and when we include those codes from the administrative data, I have lots of confidence that model performance will shoot up. Then finally, I didn't play around with the hyperparameters much, except for using a gridsearch in my models, which finds relatively decent hyperparameters. However, given some time I am sure I can eek out better performance, even on these models, with this specific data, predictors, and target variable.

For context about the diagnosis code improvement... Prior to running the below analysis, I trained the models on the ENTIRE PROMPT dataset (this included the DX_IMP columns which were the dementia diagnoses made by specialistss - which there are similar variables (DX CODES) also in admin data -- although not always made by specialists). We achieved a sensitivity of 99% and a Specificity of 99% (rounds to 100%). 

## Multinomial Logistic Regression

- Scroll past my code block if you are interested in the results.
- I am just documenting my code here for easy reference.
- I used multinomial logistic regression instead of ordinal logistic regression because the outcomes of "normal cognition, mci, dementia, and other", do not follow proportional odds.

~~~
#1. Import Libraries
print("Importing libraries...")
import pandas as pd
import numpy as np
from datetime import datetime

#Preprocessing and Modeling
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, auc, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import DataConversionWarning
import warnings

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Suppress warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
print("Libraries imported successfully.\n")

#2. Load the Dataset
print("Loading dataset...")
file_path = 'Dataset_with_overall_diag_main_Column.csv'  
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.\n")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

#3. Data Preprocessing
print("Preprocessing data...")
data['dob'] = pd.to_datetime(data['dob'], errors='coerce')

def calculate_age(born):
    today = datetime.now()
    if pd.notnull(born):
        age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        return age
    else:
        return np.nan

data['age'] = data['dob'].apply(calculate_age)
data['age'] = data['age'].fillna(data['age'].median())
print("Age variable created from 'dob'.\n")

#4. Target and Predictors
print("Defining target and predictor variables...")
target_variable = 'overall_diag_main'

#Predictor variables (I did not include any dx_imp variables)
predictor_variables = [
    'informant_relation', 'informant_relation_oth', 'live_with_informant',
    'race', 'first_language',
    'oral', 'written', 'marital_status', 'living_arrangement', 'residence_type',
    'education_level', 'education_years', 'age', 'sex',
    'cv_diseases___0', 'cv_diseases___1', 'cv_diseases___2', 'cv_diseases___3', 'cv_diseases___4',
    'cv_procedures___0', 'cv_procedures___1', 'cv_procedures___2', 'cv_procedures___3',
    'cv_procedures___4', 'cv_procedures___5', 'cv_procedures___6',
    'cereb_diseases___0', 'cereb_diseases___1', 'cereb_diseases___2', 'cereb_diseases___3',
    'cereb_diseases___4', 'cereb_diseases___5',
    'tbi___0', 'tbi___1', 'tbi___2',
    'oth_med_risk_fact___0', 'oth_med_risk_fact___1', 'oth_med_risk_fact___2',
    'oth_med_risk_fact___3', 'oth_med_risk_fact___4', 'oth_med_risk_fact___5',
    'oth_med_risk_fact___6', 'oth_med_risk_fact___7', 'oth_med_risk_fact___8',
    'oth_med_risk_fact___9', 'oth_med_risk_fact___10',
    'neuro_disorders___0', 'neuro_disorders___1', 'neuro_disorders___2', 'neuro_disorders___3',
    'neuro_disorders___4', 'neuro_disorders___5', 'neuro_disorders___6', 'neuro_disorders___7',
    'psych_diseases___11', 'psych_diseases___12', 'psych_diseases___13', 'psych_diseases___14',
    'psych_diseases___0', 'psych_diseases___1', 'psych_diseases___2', 'psych_diseases___3',
    'psych_diseases___4', 'psych_diseases___5', 'psych_diseases___6', 'psych_diseases___7',
    'psych_diseases___8', 'psych_diseases___9', 'psych_diseases___10',
    'smoking', 'alcohol', 'oth_abused_subst',
    'diagnostic_impression_complete', 'rudas_vsoltotalscore', 'rudas_praxis_score',
    'rudas_vcondtotalscore', 'rudas_judgementtraffic', 'rudas_judgementaddsafety',
    'rudas_judgementtotalscore', 'rudasmemorytotal', 'rudas_langtotalscr',
    'rudas_totallscr8', 'rowland_universal_dementia_asses',
    'moca_visuoexec', 'moca_naming', 'moca_attndigitlist', 'moca_attnletterlist',
    'moca_attnserialseven', 'moca_langrepeat', 'moca_langfluency', 'moca_abstraction',
    'moca_delayrecall', 'moca_orientation', 'moca_total', 'moca_ases_qualification',
    'moca_educationlevel', 'montreal_cognitive_assessment_mo', 'orientdate',
    'orientlocat', 'mmseregobjects', 'regattempts', 'attention_registration',
    'recall', 'langpoint', 'langspeech', 'langcommand', 'langread', 'langwrite',
    'langcopy', 'mmsetotal', 'assessorqualification', 'educationlevel_mmse',
    'mmseloc',
    'iadl_telephone_old', 'iadl_travelling_old', 'iadl_shopping_old',
    'iadl_prepmeals_old', 'iadl_housework_old', 'iadl_medicine_old',
    'iadl_managingmoney_old',
    'iadl_telephone', 'iadl_shopping', 'iadl_food', 'iadl_housework',
    'iadl_laundry', 'iadl_transportation', 'iadl_medicine', 'iadl_managingmoney'
]
if not predictor_variables:
    raise ValueError("No valid predictor variables found in the dataset.")

print(f"Number of predictor variables used: {len(predictor_variables)}\n")

#5. Including a 'prompt_id' as Identifier 
if 'prompt_id' in data.columns:
    identifier = data['prompt_id']
    print("Identifier 'prompt_id' included.\n")
else:
    identifier = pd.Series([np.nan] * len(data), name='prompt_id')
    print("Identifier 'prompt_id' not found in dataset.\n")

#6. Filter Data for Predictors and Target
print("Filtering data for predictors and target variable...")
data = data[predictor_variables + [target_variable]].copy()
data.dropna(subset=[target_variable], inplace=True)
print(f"Data shape after filtering: {data.shape}\n")

#7. Encode the Target Variable
print("Encoding the target variable...")
le = LabelEncoder()
y = le.fit_transform(data[target_variable])
target_mapping = {i: label for i, label in enumerate(le.classes_)}
print("Unique values in y after LabelEncoder:", np.unique(y))
print("Target mapping:", target_mapping, "\n")

#8. Split Data into Features and Target
print("Splitting data into features and target...")
X = data.drop(target_variable, axis=1)
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}\n")

#9. Identify Categorical and Numerical Columns
print("Identifying categorical and numerical columns...")
categorical_cols = [
    'informant_relation', 'informant_relation_oth', 'live_with_informant',
    'race', 'first_language',
    'oral', 'written', 'marital_status', 'living_arrangement', 'residence_type',
    'education_level', 'sex'
]

categorical_cols = [col for col in categorical_cols if col in X.columns]
numerical_cols = [col for col in X.columns if col not in categorical_cols]
print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}\n")

#10. Handle Data Types and Missing Values
print("Handling data types and missing values...")
for col in numerical_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

non_numeric_cols = X[numerical_cols].select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"Moving non-numeric columns to categorical_cols: {non_numeric_cols}")
    numerical_cols = [col for col in numerical_cols if col not in non_numeric_cols]
    categorical_cols.extend(non_numeric_cols)

print(f"Updated categorical columns: {categorical_cols}")
print(f"Updated numerical columns: {numerical_cols}\n")

#11. Define Preprocessing Pipelines
print("Defining preprocessing pipelines...")
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])
print("Preprocessing pipelines defined.\n")

#12. Compute Class Weights
print("Computing class weights for handling class imbalance...")
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_dict = dict(zip(np.unique(y), class_weights))
print("Class weights computed.")
print("Class weights dictionary:", class_weights_dict, "\n")

#13. Create the Multinomial Logistic Regression Model and Pipeline
print("Creating the Multinomial Logistic Regression model and pipeline...")
lr_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lr_model)
])
print("Model and pipeline created.\n")

#14. Split Data into Training and Test Sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}\n")

#15. Define Hyperparameter Distributions
print("Defining hyperparameter distributions for RandomizedSearchCV...")
param_distributions = {
    'classifier__C': np.logspace(-4, 4, 20),
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs', 'saga'],
}
print("Hyperparameter distributions defined.\n")

#16. Train the Model with Hyperparameter Tuning
print("Starting model training with hyperparameter tuning...")
random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_distributions,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)
print("Model training completed.\n")

#17. Evaluate the Model
print("Evaluating the model...")
#Best parameters
print("Best parameters found: ", random_search.best_params_)
print(f"Best cross-validation score: {random_search.best_score_:.4f}\n")

#Predict on test data
y_pred = random_search.predict(X_test)

#Classification Report
print("Classification Report:")
target_names = [str(target_mapping[i]) for i in sorted(target_mapping.keys())]
print(classification_report(y_test, y_pred, target_names=target_names))

#Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Confusion Matrix Visualization
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Accuracy and F1 Score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
print(f"Weighted F1 Score on test set: {f1:.4f}\n")

#ROC Curve and AUC (for multiclass)
print("Plotting ROC curves for each class...")
from sklearn.preprocessing import label_binarize

#Binarize the output
number_of_classes = len(np.unique(y))
y_test_binarized = label_binarize(y_test, classes=np.arange(number_of_classes))
y_score = random_search.predict_proba(X_test)

#Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(number_of_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#Plot ROC curves for each class
plt.figure(figsize=(10,8))
for i in range(number_of_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {target_names[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0,1], [0,1], 'k--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Multiclass')
plt.legend(loc='lower right')
plt.show()

#18. Save the Trained Model
print("Saving the trained model...")
import joblib
joblib.dump(random_search.best_estimator_, 'best_logistic_regression_model.pkl')
print("Model saved to 'best_logistic_regression_model.pkl'\n")

#19. Conclusion
print("All steps completed successfully.")
~~~

## XGBOOST (which is a type of GBM)

~~~
# Machine Learning Model for Predicting 'overall_diag_main' with XGBoost

# 1. Import Libraries
print("Importing libraries...")
import pandas as pd
import numpy as np
from datetime import datetime

# Preprocessing and Modeling
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, auc, f1_score
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import DataConversionWarning
import warnings

# XGBoost
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
print("Libraries imported successfully.\n")

# 2. Load the Dataset
print("Loading dataset...")
file_path = 'Dataset_with_overall_diag_main_Column.csv'  # Update with your dataset path
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.\n")
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit(1)

# 3. Data Preprocessing
print("Preprocessing data...")
data['dob'] = pd.to_datetime(data['dob'], errors='coerce')

def calculate_age(born):
    today = datetime.now()
    if pd.notnull(born):
        age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        return age
    else:
        return np.nan

data['age'] = data['dob'].apply(calculate_age)
data['age'] = data['age'].fillna(data['age'].median())
print("Age variable created from 'dob'.\n")

# 4. Define Target and Predictors
print("Defining target and predictor variables...")
target_variable = 'overall_diag_main'

predictor_variables = [
    'informant_relation', 'informant_relation_oth', 'live_with_informant',
    'race', 'first_language',
    'oral', 'written', 'marital_status', 'living_arrangement', 'residence_type',
    'education_level', 'education_years', 'age', 'sex',
    'cv_diseases___0', 'cv_diseases___1', 'cv_diseases___2', 'cv_diseases___3', 'cv_diseases___4',
    'cv_procedures___0', 'cv_procedures___1', 'cv_procedures___2', 'cv_procedures___3',
    'cv_procedures___4', 'cv_procedures___5', 'cv_procedures___6',
    'cereb_diseases___0', 'cereb_diseases___1', 'cereb_diseases___2', 'cereb_diseases___3',
    'cereb_diseases___4', 'cereb_diseases___5',
    'tbi___0', 'tbi___1', 'tbi___2',
    'oth_med_risk_fact___0', 'oth_med_risk_fact___1', 'oth_med_risk_fact___2',
    'oth_med_risk_fact___3', 'oth_med_risk_fact___4', 'oth_med_risk_fact___5',
    'oth_med_risk_fact___6', 'oth_med_risk_fact___7', 'oth_med_risk_fact___8',
    'oth_med_risk_fact___9', 'oth_med_risk_fact___10',
    'neuro_disorders___0', 'neuro_disorders___1', 'neuro_disorders___2', 'neuro_disorders___3',
    'neuro_disorders___4', 'neuro_disorders___5', 'neuro_disorders___6', 'neuro_disorders___7',
    'psych_diseases___11', 'psych_diseases___12', 'psych_diseases___13', 'psych_diseases___14',
    'psych_diseases___0', 'psych_diseases___1', 'psych_diseases___2', 'psych_diseases___3',
    'psych_diseases___4', 'psych_diseases___5', 'psych_diseases___6', 'psych_diseases___7',
    'psych_diseases___8', 'psych_diseases___9', 'psych_diseases___10',
    'smoking', 'alcohol', 'oth_abused_subst',
    'diagnostic_impression_complete', 'rudas_vsoltotalscore', 'rudas_praxis_score',
    'rudas_vcondtotalscore', 'rudas_judgementtraffic', 'rudas_judgementaddsafety',
    'rudas_judgementtotalscore', 'rudasmemorytotal', 'rudas_langtotalscr',
    'rudas_totallscr8', 'rowland_universal_dementia_asses',
    'moca_visuoexec', 'moca_naming', 'moca_attndigitlist', 'moca_attnletterlist',
    'moca_attnserialseven', 'moca_langrepeat', 'moca_langfluency', 'moca_abstraction',
    'moca_delayrecall', 'moca_orientation', 'moca_total', 'moca_ases_qualification',
    'moca_educationlevel', 'montreal_cognitive_assessment_mo', 'orientdate',
    'orientlocat', 'mmseregobjects', 'regattempts', 'attention_registration',
    'recall', 'langpoint', 'langspeech', 'langcommand', 'langread', 'langwrite',
    'langcopy', 'mmsetotal', 'assessorqualification', 'educationlevel_mmse',
    'mmseloc',
    'iadl_telephone_old', 'iadl_travelling_old', 'iadl_shopping_old',
    'iadl_prepmeals_old', 'iadl_housework_old', 'iadl_medicine_old',
    'iadl_managingmoney_old',
    'iadl_telephone', 'iadl_shopping', 'iadl_food', 'iadl_housework',
    'iadl_laundry', 'iadl_transportation', 'iadl_medicine', 'iadl_managingmoney'
]

# Ensure predictors are in the data
predictor_variables = [col for col in predictor_variables if col in data.columns]

if not predictor_variables:
    raise ValueError("No valid predictor variables found in the dataset.")

print(f"Number of predictor variables used: {len(predictor_variables)}\n")

# 5. Include 'prompt_id' as Identifier (if needed)
if 'prompt_id' in data.columns:
    identifier = data['prompt_id']
    print("Identifier 'prompt_id' included.\n")
else:
    identifier = pd.Series([np.nan] * len(data), name='prompt_id')
    print("Identifier 'prompt_id' not found in dataset.\n")

# 6. Filter Data for Predictors and Target
print("Filtering data for predictors and target variable...")
data = data[predictor_variables + [target_variable]].copy()
data.dropna(subset=[target_variable], inplace=True)
print(f"Data shape after filtering: {data.shape}\n")

# 7. Encode the Target Variable
print("Encoding the target variable...")
# Adjust target variable to ensure labels start from 0
y_original = data[target_variable].astype(str)  # Convert to string in case of non-integer labels
label_mapping = {label: idx for idx, label in enumerate(sorted(y_original.unique()))}
y = y_original.map(label_mapping).astype(int)
target_mapping = {v: k for k, v in label_mapping.items()}
print("Unique values in y after encoding:", np.unique(y))
print("Target mapping:", target_mapping, "\n")

# 8. Split Data into Features and Target
print("Splitting data into features and target...")
X = data.drop(target_variable, axis=1)
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}\n")

# 9. Identify Categorical and Numerical Columns
print("Identifying categorical and numerical columns...")
categorical_cols = [
    'informant_relation', 'informant_relation_oth', 'live_with_informant',
    'race', 'first_language',
    'oral', 'written', 'marital_status', 'living_arrangement', 'residence_type',
    'education_level', 'sex',
    # Add other known categorical columns if needed
]
categorical_cols = [col for col in categorical_cols if col in X.columns]
numerical_cols = [col for col in X.columns if col not in categorical_cols]
print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}\n")

# 10. Handle Data Types and Missing Values
print("Handling data types and missing values...")
for col in numerical_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

non_numeric_cols = X[numerical_cols].select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"Moving non-numeric columns to categorical_cols: {non_numeric_cols}")
    numerical_cols = [col for col in numerical_cols if col not in non_numeric_cols]
    categorical_cols.extend(non_numeric_cols)

print(f"Updated categorical columns: {categorical_cols}")
print(f"Updated numerical columns: {numerical_cols}\n")

# 11. Define Preprocessing Pipelines
print("Defining preprocessing pipelines...")
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])
print("Preprocessing pipelines defined.\n")


# 12. Compute Sample Weights for Class Imbalance
print("Computing sample weights for handling class imbalance...")
sample_weights = compute_sample_weight(class_weight='balanced', y=y)
print("Sample weights computed.\n")

# 13. Create the XGBoost Model and Pipeline
print("Creating the XGBoost model and pipeline...")
number_of_classes = len(np.unique(y))
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=number_of_classes,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1,
    use_label_encoder=False
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])
print("Model and pipeline created.\n")

# 14. Split Data into Training and Test Sets
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weights, test_size=0.2, stratify=y, random_state=42
)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}\n")

# 15. Define Hyperparameter Distributions
print("Defining hyperparameter distributions for RandomizedSearchCV...")
param_distributions = {
    'classifier__n_estimators': [100, 200, 300, 500],
    'classifier__max_depth': [3, 5, 7, 10],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.6, 0.8, 1.0],
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],
    'classifier__gamma': [0, 0.1, 0.2],
    'classifier__reg_lambda': [0.5, 1, 1.5],
    'classifier__min_child_weight': [1, 5, 10],
}
print("Hyperparameter distributions defined.\n")

# 16. Train the Model with Hyperparameter Tuning
print("Starting model training with hyperparameter tuning...")
random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_distributions,
    n_iter=100,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Define a custom fit_params dictionary to pass sample weights
fit_params = {'classifier__sample_weight': sw_train}

random_search.fit(X_train, y_train, **fit_params)
print("Model training completed.\n")

# 17. Evaluate the Model
print("Evaluating the model...")
# Best parameters
print("Best parameters found: ", random_search.best_params_)
print(f"Best cross-validation score: {random_search.best_score_:.4f}\n")

# Predict on test data
y_pred = random_search.predict(X_test)

# Classification Report
print("Classification Report:")
target_names = [target_mapping[i] for i in sorted(target_mapping.keys())]
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Confusion Matrix Visualization
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Accuracy and F1 Score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
print(f"Weighted F1 Score on test set: {f1:.4f}\n")

# ROC Curve and AUC (for multiclass)
print("Plotting ROC curves for each class...")
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Binarize the output
y_test_binarized = label_binarize(y_test, classes=np.arange(number_of_classes))
y_score = random_search.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(number_of_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(10,8))
for i in range(number_of_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {target_names[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0,1], [0,1], 'k--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Multiclass')
plt.legend(loc='lower right')
plt.show()

# 18. Extract Feature Importances
print("Extracting feature importances...")
best_model = random_search.best_estimator_

# Get feature names from preprocessing
onehot = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
onehot_columns = list(onehot.get_feature_names_out(categorical_cols))
feature_names = numerical_cols + onehot_columns

importances = best_model.named_steps['classifier'].feature_importances_

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 20 Features by Importance:")
print(feature_importances.head(20))

# Feature Importances Plot
plt.figure(figsize=(10,8))
sns.barplot(
    x='Importance', y='Feature',
    data=feature_importances.head(20),
    palette='viridis'
)
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 19. Save the Trained Model
print("Saving the trained model...")
import joblib
joblib.dump(random_search.best_estimator_, 'best_xgboost_model.pkl')
print("Model saved to 'best_xgboost_model.pkl'\n")

# 20. Conclusion
print("All steps completed successfully.")
~~~

MULTINOMIAL

**Accuracy on test set: 53.68%**
Weighted F1 Score on test set: 0.5413

Best parameters found:  {'classifier__solver': 'lbfgs', 'classifier__penalty': 'l2', 'classifier__C': 0.004832930238571752}
Best cross-validation score: 0.5559

**Classification Report:**
              precision    recall  f1-score   support

           1       0.35      0.70      0.47        44
           2       0.57      0.51      0.54       127
           3       0.58      0.58      0.58        97
           4       0.65      0.46      0.54       112

    accuracy                           0.54       380
   macro avg       0.54      0.56      0.53       380
weighted avg       0.57      0.54      0.54       380











## RANDOM FOREST
Accuracy on test set: 58.16%
Weighted F1 Score on test set: 0.5712

Top 20 Features by Importance:
                           Feature  Importance
97                       mmsetotal    0.057823
81                      moca_total    0.054470
1                              age    0.045069
79                moca_delayrecall    0.036899
90                          recall    0.032810
60  diagnostic_impression_complete    0.027678
0                  education_years    0.023684
71                  moca_visuoexec    0.022399
85                      orientdate    0.020168
80                moca_orientation    0.019783
86                     orientlocat    0.017861
7                cv_procedures___0    0.017055
57                         smoking    0.016015
58                         alcohol    0.015770
14              cereb_diseases___0    0.014995
34             neuro_disorders___0    0.014730
99             educationlevel_mmse    0.014048
83             moca_educationlevel    0.013811
89          attention_registration    0.012644
20                         tbi___0    0.012380


# XGBOOST


Accuracy on test set: 58.42%
Weighted F1 Score on test set: 0.5873

Classification Report:
              precision    recall  f1-score   support

           1       0.42      0.66      0.51        44
           2       0.55      0.63      0.59       127
           3       0.58      0.64      0.61        97
           4       0.88      0.46      0.60       112

    accuracy                           0.58       380
   macro avg       0.61      0.60      0.58       380
weighted avg       0.64      0.58      0.59       380

Confusion Matrix:
[[29 11  0  4]
 [16 80 28  3]
 [ 6 29 62  0]
 [18 26 17 51]]

Top 20 Features by Importance:
                              Feature  Importance
97                          mmsetotal    0.031960
7                   cv_procedures___0    0.030082
14                 cereb_diseases___0    0.020586
113               iadl_transportation    0.019377
81                         moca_total    0.015681
90                             recall    0.015647
60     diagnostic_impression_complete    0.013398
20                            tbi___0    0.012016
84   montreal_cognitive_assessment_mo    0.011488
108                    iadl_telephone    0.011065
34                neuro_disorders___0    0.011045
107            iadl_managingmoney_old    0.010589
79                   moca_delayrecall    0.010017
189               education_level_5.0    0.009771
3                     cv_diseases___1    0.009673
80                   moca_orientation    0.009604
85                         orientdate    0.009492
179                residence_type_0.0    0.008973
2                     cv_diseases___0    0.008965
181                residence_type_2.0    0.008640









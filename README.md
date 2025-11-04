# Predicting Airline Customer Satisfaction Using Decision Trees and Random Forest
**Project Overview**
This project applies machine learning techniques to predict airline customer satisfaction using survey responses from 129,880 passengers. The dataset includes various features related to the flight experience, such as travel class, flight distance, customer type, and in-flight services.

The goal is to build and evaluate models capable of classifying customers as satisfied or dissatisfied with their flight experience. Three models were developed and compared:

1: **Baseline Decision Tree** – a simple model providing a starting point for comparison.

2: **Tuned Decision Tree** – optimised using GridSearchCV to improve accuracy, precision, recall, and F1-score.

3: **Random Forest Classifier** – an ensemble model used to enhance performance and robustness.

T**he workflow involved:**

- **Data Exploration and Cleaning**: Handling missing values, encoding categorical variables, and checking data distributions.

- **Model Development**: Training decision tree and random forest models using Scikit-learn.

- **Hyperparameter Tuning**: Using GridSearchCV to identify the best parameter combinations for each model.

- **Evaluation**: Assessing performance through accuracy, precision, recall, F1-score, and confusion matrices.

Results demonstrated that the random forest model significantly outperformed the tuned decision tree, offering higher predictive accuracy and better generalisation. This finding reinforces the effectiveness of ensemble methods for classification tasks in customer satisfaction analytics.

The project illustrates how airlines can leverage predictive modelling to anticipate customer satisfaction levels, enabling data-driven strategies for improving passenger experience.

**Importing libraries for data processing and machine learning**

```
# Standard operational package imports
import pandas as pd
import numpy as np

# Important imports for modelling and evaluation
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
# Visualisation package imports
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
```
Load the airline customer satisfaction dataset and display the first 10 rows
```
df_original = pd.read_csv("/content/Invistico_Airline.csv")
df_original.head(10)
```
<img width="1277" height="594" alt="image" src="https://github.com/user-attachments/assets/b90ce856-85da-4153-bbee-b4047587e154" />

Check data types, unique values in the Class column, and distribution of satisfaction levels
```
df_original.dtypes
df_original.Class.unique()
round(df_original.satisfaction.value_counts(normalize = True)*100,2)
```
<img src="https://github.com/user-attachments/assets/9a16dddb-598b-4bd1-bd6c-7a75ad20e0da" width="250">

<p>Text below the first image.</p>

<img src="https://github.com/user-attachments/assets/06d472f8-8230-46a1-962e-f36672f74a95" width="250">

<p>Text below the second image.</p>

<img src="https://github.com/user-attachments/assets/f56a59b2-c651-4389-a823-b6c174308f97" width="250">

<p>Text below the third image.</p>

## Data Cleaning and Preprocessing

Let us check for missing values and show the size. 
```
df_original.isnull().sum()
```
<img width="297" height="732" alt="image" src="https://github.com/user-attachments/assets/4ca6d5f6-14da-47a3-921d-f3da0f71bdf0" />

Let us remove rows with the missing values 
```
df_subset = df_original.dropna(axis = 0)
df_subset.reset_index(drop = True, inplace = True)
```
Encode categorical variables
```
df_subset = pd.get_dummies(df_subset, columns = ['Class', 'Customer Type', 'Type of Travel'])
```

## Base Decision Tree Model
Split data into training and testing sets

```
y = df_subset['satisfaction'].map({"satisfied": 1, "dissatisfied": 0})
X = df_subset.drop('satisfaction', axis = 1)
# Split the variables into dependent and independent.
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```
Train the initial decision tree model and evaluate performance metrics
```
decision_tree = DecisionTreeClassifier(random_state = 0)
decision_tree.fit(X_train, y_train)
dt_pred = decision_tree.predict(X_test)

print(f'Accuracy Score  : {accuracy_score(y_test, dt_pred):.4f}')
print(f'Precision Score : {precision_score(y_test, dt_pred):.4f}')
print(f'Recall Score    : {recall_score(y_test, dt_pred):.4f}')
print(f'F1 Score        : {f1_score(y_test, dt_pred):.4f}')
```
<img width="225" height="70" alt="image" src="https://github.com/user-attachments/assets/c6bb91e0-81ad-4e2d-835d-ac6f9d56d2d6" />


Create and display a confusion matrix for the decision tree model
```
cm = confusion_matrix(y_test, dt_pred, labels = decision_tree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = decision_tree.classes_)
disp.plot(values_format = '')
```
<img width="561" height="436" alt="image" src="https://github.com/user-attachments/assets/c185eee1-c4af-4130-833c-80cf596aa466" />

Visualise the decision tree structure (limited to depth 2 for readability)

<img width="1285" height="695" alt="image" src="https://github.com/user-attachments/assets/dff9d9ca-bd1a-470c-9f38-8c069802d3a6" />


## Hyperparameter Tuning for Decision Tree

Define parameter grid and scoring metrics for hyperparameter tuning.

```
tree_para = {'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50],
             'min_samples_leaf': [2,3,4,5,6,7,8,9, 10, 15, 20, 50]}
scoring = {'accuracy': 'accuracy', 'precision': 'precision',
           'recall': 'recall', 'f1': 'f1'}
```
Perform grid search to find optimal hyperparameters for decision tree
```
tuned_decision_tree = DecisionTreeClassifier(random_state = 0)


clf = GridSearchCV(tuned_decision_tree,
                  tree_para,
                  scoring = scoring,
                  cv = 5,
                  refit = 'f1',
                  n_jobs=-1)
clf.fit(X_train,y_train)
```
<img width="688" height="194" alt="image" src="https://github.com/user-attachments/assets/c1e7bb4e-6bca-477f-b289-0bfbf2761a65" />


Display the best estimator and its validation score

```
print(f'Best Average Validation Score: {clf.best_score_:.4f}')
```

Best Average Validation Score: 0.9454

Create a results summary for the tuned decision tree

```
best_results = pd.DataFrame({
    'Model': ['Tuned Decision Tree'],
    'F1': [clf.best_score_],
    'Recall': [clf.cv_results_['mean_test_recall'][clf.best_index_]],
    'Precision': [clf.cv_results_['mean_test_precision'][clf.best_index_]],
    'Accuracy': [clf.cv_results_['mean_test_accuracy'][clf.best_index_]]
})
best_results
```
<img width="501" height="68" alt="image" src="https://github.com/user-attachments/assets/dc26ed08-b344-48bd-80c2-760496a4cdda" />

Now let us check its confusion matrix:

```
y_pred_tuned = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_tuned, labels = clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot(values_format = '')
```
<img width="547" height="436" alt="image" src="https://github.com/user-attachments/assets/21153a0e-f16d-4da7-be14-51c5f4d26a6a" />

You can find this file in the repository

```
import pickle
with open('tuned_dt.pkl', 'wb') as file:
  pickle.dump(clf, file)
```
## Random Forest Model


First, we need to split the validation data from train data.

```
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.25, stratify= y_train, random_state = 42)
```
Define an extensive parameter grid for random forest hyperparameter tuning
```
cv_params = {
    'n_estimators': [100, 150],
    'max_depth': [15, 25, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 0.5],
    'max_samples': [0.8, None]
}
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}
```
Set up custom validation split and perform grid search for random forest

```
custom_split = [0 if x in X_val.index else -1 for x in X_train.index]
custom_split = PredefinedSplit(custom_split)

rf = RandomForestClassifier(random_state = 0)

rf_clf = GridSearchCV(estimator = rf,
                   param_grid = cv_params,
                   scoring = scoring,
                   cv = custom_split,
                   refit = 'f1',
                  n_jobs = -1,
                  verbose = 1)

rf_clf.fit(X_train, y_train)
```
<img width="719" height="207" alt="image" src="https://github.com/user-attachments/assets/17719794-c4f0-4c43-9498-10450092c8c6" />

Save the tuned random forest model. You can find it in the repository

```
with open('tuned_rf.pkl', 'wb') as file:
    pickle.dump(rf_clf, file)
```
Train the optimised random forest model with best parameters

```
rf_opt = RandomForestClassifier(max_depth = None,
                                max_features = 0.5,
                                max_samples = None,
                                min_samples_leaf = 1,
                                min_samples_split= 2,
                                n_estimators = 150,
                                random_state = 0
                               )

rf_opt.fit(X_train, y_train)
y_pred = rf_opt.predict(X_test)
```
Calculate performance metrics for the optimised random forest and compare it with the tuned decision tree model to see which one is performing better.

```
precision = precision_score(y_test, y_pred)
recall  = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
comparison_table = pd.DataFrame({
    'Model': ['Base Decision Tree', 'Tuned Decision Tree', 'Tuned Random Forest'],
    'F1': [0.9411, clf.best_score_, f1],
    'Recall': [0.9398, clf.cv_results_['mean_test_recall'][clf.best_index_], recall],
    'Precision': [0.9424, clf.cv_results_['mean_test_precision'][clf.best_index_], precision],
    'Accuracy': [0.9356, clf.cv_results_['mean_test_accuracy'][clf.best_index_], accuracy]
})
```
<img width="505" height="142" alt="image" src="https://github.com/user-attachments/assets/0f4d78bc-7c4a-4dde-b66e-c3a4febaf30f" />


Create and display a confusion matrix for the random forest model
```
cm = confusion_matrix(y_test, y_pred, labels = rf_opt.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = rf_opt.classes_)
disp.plot(values_format = '')
```

<img width="539" height="434" alt="image" src="https://github.com/user-attachments/assets/f49cf31d-bb8a-4ecf-8755-073ae115d9f8" />





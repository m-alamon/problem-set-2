'''
Name: Matthew Alamon
PART 4: Decision Trees
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score

def load_data():
    """
    Loads the datasets from CSV files generated from Part 3.
    """
    df_arrests_train = pd.read_csv('data/df_arrests_train.csv')
    df_arrests_test = pd.read_csv('data/df_arrests_test.csv')
    return df_arrests_train, df_arrests_test

def decision_tree(df_arrests_train, df_arrests_test):
    """
    Runs decision tree classification with hyperparameter tuning.
    """
    # Separating features and target variable from the training data.
    X_train = df_arrests_train[['num_fel_arrests_last_year']]
    y_train = df_arrests_train['y']
    
    X_test = df_arrests_test[['num_fel_arrests_last_year']]
    y_test = df_arrests_test['y']
    
    # Initializes the Decision Tree model.
    dt_model = DTC()

    # Creates a parameter grid for tree depth.
    param_grid_dt = {
        'max_depth': [3, 5, 7]
    }

    # Initializes GridSearchCV.
    gs_cv_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=5, scoring='accuracy')

    # Fitting the model.
    gs_cv_dt.fit(X_train, y_train)
    
    # Finding the optimal value for max_depth.
    best_max_depth = gs_cv_dt.best_params_['max_depth']
    print(f"Optimal value for max_depth: {best_max_depth}")

    # Predicting for the test set.
    df_arrests_test['pred_dt'] = gs_cv_dt.predict(X_test)

    # Saves predictions to CSV.
    df_arrests_test.to_csv('data/df_arrests_test_with_predictions.csv', index=False)
    print("Dataframe saved as df_arrests_test_with_predictions.csv.")

    # Printing the accuracy on the test set.
    accuracy = accuracy_score(y_test, df_arrests_test['pred_dt'])
    print(f"Accuracy on the test set: {accuracy:.2%}")

def main():
    df_arrests_train, df_arrests_test = load_data()
    decision_tree(df_arrests_train, df_arrests_test)

if __name__ == "__main__":
    main()

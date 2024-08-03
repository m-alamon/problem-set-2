'''
Name: Matthew Alamon
PART 3: Logistic Regression
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold as KFold_str

def load_data():
    """
    Loads the dataset from CSV generated from Part 2.
    """
    df_arrests = pd.read_csv('data/df_arrests.csv')
    return df_arrests

def logistic_regression(df_arrests):
    """
    Runs logistic regression with hyperparameter tuning.
    """
    # Splits the data into training and testing sets.
    df_arrests_train, df_arrests_test = train_test_split(
        df_arrests,
        test_size=0.3,
        shuffle=True,
        stratify=df_arrests['y'],
        random_state=42  # Ensures reproducibility of results.
    )

    # Defines the features and target.
    features = ['num_fel_arrests_last_year']
    target = 'y'

    # Defines the parameter grid for GridSearchCV.
    param_grid = {'C': [0.01, 0.1, 1]}

    # Initializes the Logistic Regression model.
    lr_model = LogisticRegression(max_iter=1000)

    # Initializes GridSearchCV.
    gs_cv = GridSearchCV(
        estimator=lr_model,
        param_grid=param_grid,
        cv=KFold_str(n_splits=5),  # 5-fold cross-validation
        scoring='accuracy',
        n_jobs=-1  # Using all available cores.
    )

    # Fitting the model.
    gs_cv.fit(df_arrests_train[features], df_arrests_train[target])

    # Getting the best parameter and its score.
    best_C = gs_cv.best_params_['C']
    best_score = gs_cv.best_score_

    # Prints optimal value for C and its regularization.
    print(f"Optimal value for C: {best_C}")
    if best_C == max(param_grid['C']):
        print("This has the least regularization.")
    elif best_C == min(param_grid['C']):
        print("This has the most regularization.")
    else:
        print("This is in the middle.")

    # Predicting on the test set.
    df_arrests_test['pred_lr'] = gs_cv.predict(df_arrests_test[features])

    # Saves the resulting DataFrames to CSV.
    df_arrests_train.to_csv('data/df_arrests_train.csv', index=False)
    df_arrests_test.to_csv('data/df_arrests_test.csv', index=False)

    # Printing the classification report on the test set.
    print(classification_report(df_arrests_test[target], df_arrests_test['pred_lr']))

    print("DataFrames saved as df_arrests_train.csv and df_arrests_test.csv.")

def main():
    df_arrests = load_data()
    logistic_regression(df_arrests)

if __name__ == "__main__":
    main()

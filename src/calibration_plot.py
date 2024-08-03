'''
Name: Matthew Alamon
PART 5: Calibration-light
'''

# Import any further packages you may need for PART 5
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """
    Loads the datasets from CSV files generated from Part 3 & 4.
    """
    df_arrests_test = pd.read_csv('data/df_arrests_test.csv')
    df_arrests_tests_with_predictions = pd.read_csv('data/df_arrests_test_with_predictions.csv')
    return df_arrests_test, df_arrests_tests_with_predictions

def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.
    
    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.
    """
    # Calculates calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Creating the Seaborn plot.
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def compute_ppv_and_auc(df_arrests_test, df_arrests_tests_with_predictions):
    """
    Compute PPV and AUC for the models.
    """
    # Extracting predictions.
    decision_tree_preds = df_arrests_tests_with_predictions['pred_dt']  # Probabilities from the decision tree
    logistic_regression_preds = df_arrests_test['pred_lr']  # Probabilities from the logistic regression model

    # Extracting true labels.
    y_true = df_arrests_test['y']

    # Plotting calibration curves.
    print("Calibration plot for Logistic Regression:")
    calibration_plot(y_true, logistic_regression_preds, n_bins=5)

    print("Calibration plot for Decision Tree:")
    calibration_plot(y_true, decision_tree_preds, n_bins=5)

    # Calculating the top 50 predicted risks for PPV calculation.
    top_50_logistic_indices = logistic_regression_preds.nlargest(50).index
    top_50_decision_tree_indices = decision_tree_preds.nlargest(50).index

    # Calculating PPV (Positive Predictive Value).
    ppv_logistic = (y_true.loc[top_50_logistic_indices].sum()) / len(top_50_logistic_indices)
    ppv_decision_tree = (y_true.loc[top_50_decision_tree_indices].sum()) / len(top_50_decision_tree_indices)

    print(f"PPV for Logistic Regression (top 50 predicted risk): {ppv_logistic:.4f}")
    print(f"PPV for Decision Tree (top 50 predicted risk): {ppv_decision_tree:.4f}")

    # Calculating AUC.
    auc_logistic = roc_auc_score(y_true, logistic_regression_preds)
    auc_decision_tree = roc_auc_score(y_true, decision_tree_preds)

    print(f"AUC for Logistic Regression: {auc_logistic:.4f}")
    print(f"AUC for Decision Tree: {auc_decision_tree:.4f}")

    # Determines which model is better calibrated.
    print("Which model is more calibrated?")
    if auc_logistic > auc_decision_tree:
        print("Logistic Regression is more accurate according to AUC.")
    else:
        print("Decision Tree is more accurate according to AUC.")

def main():
    df_arrests_test, df_arrests_tests_with_predictions = load_data()
    compute_ppv_and_auc(df_arrests_test, df_arrests_tests_with_predictions)

if __name__ == "__main__":
    main()

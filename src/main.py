'''
Name: Matthew Alamon
Class: INST414
Section: WB21
Assignment: Problem Set 2
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


def main():
    # PART 1: Instantiate etl, saving the two datasets in `./data/`
    etl.main()
    
    # PART 2: Call functions/instantiate objects from preprocessing
    preprocessing.main()

    # PART 3: Call functions/instantiate objects from logistic_regression
    logistic_regression.main()

    # PART 4: Call functions/instantiate objects from decision_tree
    decision_tree.main()

    # PART 5: Call functions/instantiate objects from calibration_plot
    calibration_plot.main()

if __name__ == "__main__":
    main()

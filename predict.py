from linear_regression import linear_regression_predict
from polynominal_regression import polynomial_regression_predict
from random_forest import randomforest_predict
from feature_selection import feature_selection
from deep_learning_nn import bp_predict
from svr import svr_predict


def main():
    feature_selection()
    linear_regression_predict()
    polynomial_regression_predict()
    randomforest_predict()
    bp_predict()
    svr_predict()


if __name__ == "__main__":
    main()

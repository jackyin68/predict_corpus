import linear_regression, polynominal_regression, random_forest,feature_selection


def main():
    feature_selection.feature_selection()
    linear_regression.linear_regression_predict()
    polynominal_regression.polynomial_regression_predict()
    random_forest.randomforest_predict()


if __name__ == "__main__":
    main()

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import tree, metrics
import pydotplus


def randomforest_predict():
    warnings.filterwarnings('ignore')

    df_data = pd.read_csv("data/housing.data", delim_whitespace=True)
    X = df_data.drop(["MEDV"], axis=1)
    y = df_data["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=128)

    param_grid = {
        'n_estimators': [5, 10, 20, 50, 100, 200],  # tree number
        'max_depth': [3, 5, 7],  # max depth
        'max_features': [0.6, 0.7, 0.8, 1]  # max features
    }

    rf = RandomForestRegressor()
    grid = GridSearchCV(rf, param_grid=param_grid, cv=3)
    grid.fit(X_train, y_train)
    print("best_params", grid.best_params_)

    rf_reg = grid.best_estimator_
    print(rf_reg)

    estimator = rf_reg.estimators_[3]
    dot_data = tree.export_graphviz(estimator, out_file=None, filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("result/rf_reg.png")

    feature_names = X.columns
    feature_importances = rf_reg.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    for index in indices:
        print("feature %s (%f)" % (feature_names[index], feature_importances[index]))

    plt.figure(figsize=(16, 8))
    plt.title("feature importance of random forest")
    plt.bar(range(len(feature_importances)), feature_importances[indices], color='b')
    plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices], color='b')
    plt.show()

    rst = {"label": y_test, "prediction": rf_reg.predict(X_test)}
    rst = pd.DataFrame(rst)
    print(rst.head())

    rst['label'].plot(style='k.', figsize=(15, 5))
    rst['prediction'].plot(style='r.')
    plt.legend(fontsize=15, markerscale=3)
    plt.tick_params(labelsize=25)
    plt.grid()
    plt.show()

    MSE = metrics.mean_squared_error(y, rf_reg.predict(X))
    print(np.sqrt(MSE))

    submission = {"prediction": rf_reg.predict(X_test)}
    submission = pd.DataFrame(submission)
    submission.to_csv("result/price_predict_randomforest.csv")

    y_predict = rf_reg.predict(X_test)
    x_data = pd.Series(range(len(y_test)))[:, np.newaxis]
    y_test_data = y_test[:, np.newaxis]
    y_predict_data = y_predict[:, np.newaxis]
    plt.plot(x_data, y_test_data, label='Price')
    plt.plot(x_data, y_predict_data, label='Predict price')
    plt.xlabel('Entity')
    plt.ylabel('Price')
    plt.title('Price prediction (random forest)')
    plt.legend()
    plt.savefig('result/price_predict_random_forest.png')
    plt.show()
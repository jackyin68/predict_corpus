import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    return pipeline


def polynomial_regression_predict():
    warnings.filterwarnings('ignore')

    df = pd.read_csv("data/housing.data", delim_whitespace=True)
    # X = df.drop(["MEDV"], axis=1)
    cols = ['CRM', 'INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT']
    X = df[cols]
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    model = polynomial_model(degree=3)
    start = time.clock()
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    cv_score = model.score(X_test, y_test)
    print(
        "elaspe: {0:.6f}; train_score: {1:0.6f}; cv_score: {2:.6f}".format(time.clock() - start, train_score, cv_score))

    y_predict = model.predict(X_test)
    deviation = y_predict - y_test
    print("deviation:", deviation)
    RMSE = np.sum(np.sqrt(deviation * deviation)) / 102
    print("RMSE:", RMSE)

    rst = {'prediction': y_predict}
    rst_df = pd.DataFrame(rst)
    rst_df.to_csv("result/poly_predict.csv")

    rst = {"label": y_test, "prediction": y_predict}
    rst = pd.DataFrame(rst)
    rst['label'].plot(style='k.', figsize=(15, 5))
    rst['prediction'].plot(style='r.')
    plt.legend(fontsize=15, markerscale=3)
    plt.tick_params(labelsize=25)
    plt.grid()
    plt.savefig('result/price_predict_grid_poly.png')
    plt.show()

    x_data = pd.Series(range(len(y_test)))[:, np.newaxis]
    y_test_data = y_test[:, np.newaxis]
    y_predict_data = y_predict[:, np.newaxis]
    plt.plot(x_data, y_test_data, label='Price')
    plt.plot(x_data, y_predict_data, label='Predict price')
    plt.xlabel('Entity')
    plt.ylabel('Price')
    plt.title('Price prediction (Poly)')
    plt.legend()
    plt.savefig('result/price_predict_poly.png')
    plt.show()

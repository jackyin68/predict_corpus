import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def linear_regression_predict():
    warnings.filterwarnings('ignore')

    df = pd.read_csv("data/housing.data", delim_whitespace=True)
    X = df.drop(["MEDV"], axis=1)
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=128)
    model = LinearRegression()
    model.fit(X_train, y_train)

    model_score = model.score(X_test, y_test)
    print("model_score", model_score)
    plt.plot(model.coef_)
    plt.show()

    y_predict = model.predict(X_test)
    deviation = y_predict - y_test
    print("deviation:", deviation)
    rmse = np.sum(np.sqrt(deviation * deviation)) / 102
    print("RMSE:", rmse)

    rst = {'prediction': y_predict}
    rst_df = pd.DataFrame(rst)
    rst_df.to_csv("result/price_predict_lr.csv")

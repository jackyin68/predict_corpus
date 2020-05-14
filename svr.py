import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
confirmed = []


def get_data(filename):
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            dates.append(int(row[0].split('-')[0]))
            confirmed.append(float(row[1]))
    return


def predict_confirmed(dates, confirmed, x):
    dates = np.reshape(dates, (len(dates), 1))
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=3)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin.fit(dates, confirmed)
    svr_poly.fit(dates, confirmed)
    svr_rbf.fit(dates, confirmed)

    plt.scatter(dates, confirmed, color='black', label='Data', linewidths=1)
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Confirmed')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.savefig('result/covid-svr.png')
    plt.show()

    x = np.reshape(x, (len(x), 1))
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


def svr_predict():
    get_data('./data/cn_svr.csv')
    predict_confirm = predict_confirmed(dates, confirmed, [29])
    print(predict_confirm)

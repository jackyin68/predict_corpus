from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def bp_predict():
    warnings.filterwarnings('ignore')

    df = pd.read_csv("data/housing.data", delim_whitespace=True)
    X = df.drop(["MEDV"], axis=1)
    # cols = ['CRM', 'INDUS', 'NOX', 'RM', 'TAX', 'PTRATIO', 'LSTAT']
    # X = df[cols]
    y = df["MEDV"]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.15, random_state=128)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train, y_train = scaler.fit_transform(X_train_raw)[:, np.newaxis, :], scaler.fit_transform(
        pd.DataFrame(y_train_raw))
    X_test, y_test = scaler.fit_transform(X_test_raw)[:, np.newaxis, :], scaler.fit_transform(pd.DataFrame(y_test_raw))

    epochs = 1000
    batch_size = 20
    model = Sequential()
    model.add(LSTM(units=30, return_sequences=True, input_dim=X_train.shape[-1], input_length=X_train.shape[1]))
    # model.add(LSTM(units=30, return_sequences=True))
    # model.add(LSTM(units=30, return_sequences=True))
    # model.add(LSTM(units=30, return_sequences=True))
    # model.add(LSTM(units=10, return_sequences=True))
    model.add(LSTM(units=10))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    medv = model.predict(X_test)
    scaler.fit_transform(pd.DataFrame(y_test_raw))
    medv = scaler.inverse_transform(medv)
    y_test = np.array(y_test_raw)[:, np.newaxis]
    rms = np.sqrt(np.mean(np.power((y_test - medv), 2)))
    print(rms)

    plt.figure(figsize=(16, 8))
    dict_data = {
        'price': y_test[..., 0],
        'prediction': medv[..., 0]
    }
    data_pd = pd.DataFrame(dict_data)
    plt.plot(data_pd[['price']], label='Price')
    plt.plot(data_pd[['prediction']], label='Predict price')
    plt.xlabel('Sequence')
    plt.ylabel('Housing Price')
    plt.title('Housing price prediction')
    plt.legend()
    plt.savefig('result/house_price_predict.png')
    plt.show()

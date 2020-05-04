import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler


def feature_selection():
    warnings.filterwarnings('ignore')

    df = pd.read_csv("data/housing.data", delim_whitespace=True)
    drop_columns = ['MEDV']
    x = df.drop(drop_columns, axis=1)
    y = df['MEDV']

    select_kbest = SelectKBest(f_regression, k=5)
    x_new = select_kbest.fit_transform(x, y)  # .get_support(indices=True)
    x_index = x.columns[select_kbest.get_support()]
    x_df = pd.DataFrame(x_new, columns=x_index)
    pd.plotting.scatter_matrix(x_df, alpha=0.5, figsize=(12, 12))
    plt.savefig('result/feature_selection.png')
    plt.show()

    scaler = MinMaxScaler()
    for feature in df.columns:
        df['d_' + feature] = scaler.fit_transform(df[[feature]])
        print(df.head())

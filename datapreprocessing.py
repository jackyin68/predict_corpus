import pandas as pd


data = pd.read_csv("./data/中国_cov.csv")
data = data[['date', 'confirmed']]
data['date'] = pd.to_datetime(data['date'])
data.index = data['date']
data = data.sort_index()

# total = data[1].cumsum()
total = data[['confirmed']] - data[['confirmed']].shift(1)
total = total.reset_index()['confirmed']
total[1:].to_csv('./data/cn_svr.csv')
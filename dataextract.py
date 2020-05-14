# encoding=utf8
import csv
import os
import time
import pandas as pd

country = '中国'
rawdata_root = os.getcwd() + os.sep + "rawdata" + os.sep
data_root = os.getcwd() + os.sep + "data" + os.sep + country

df_num = 0
for file in os.listdir(rawdata_root):
    dfs = pd.read_csv(rawdata_root + file)
    if df_num == 0:
        df = dfs[(dfs['country'] == country) & (dfs['province'].isnull())][['date', 'country', 'confirmed', 'cured']]
    else:
        df2 = dfs[(dfs['country'] == country) & (dfs['province'].isnull())][['date', 'country', 'confirmed', 'cured']]
        df = df.append(df2, ignore_index=True)
    df_num = df_num + 1

df.to_csv(data_root + '_cov.csv', index=0)
print(country, " Data generated ...")

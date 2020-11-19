from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn import preprocessing, model_selection, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pytz import timezone
import pytz
import pprint as pp
import itertools

import copy
def combinations(target,data):
    for i in range(len(data)):
        new_target = copy.copy(target)
        new_data = copy.copy(data)
        new_target.append(data[i])
        new_data = data[i+1:]
        combinations(new_target,
                     new_data)


data = pd.read_csv("data/SolarPrediction.csv")

# remove outlier
data = data.drop(6465)
data = data.sort_values(['UNIXTime'], ascending = [True])

column = data["UNIXTime"]
max_value = column.max()
min_value = column.min()
#print(data.loc[data['UNIXTime'] == min_value])
#print(data.loc[data['UNIXTime'] == max_value])

hawaii= timezone('Pacific/Honolulu')
data.index =  pd.to_datetime(data['UNIXTime'], unit='s')
data.index = data.index.tz_localize(pytz.utc).tz_convert(hawaii)
data['MonthOfYear'] = data.index.strftime('%m').astype(int)
data['DayOfYear'] = data.index.strftime('%j').astype(int)
data['WeekOfYear'] = data.index.strftime('%U').astype(int)
data['TimeOfDay(h)'] = data.index.hour
data['TimeOfDay(m)'] = data.index.hour*60 + data.index.minute
data['TimeOfDay(s)'] = data.index.hour*60*60 + data.index.minute*60 + data.index.second
data['TimeSunRise'] = pd.to_datetime(data['TimeSunRise'], format='%H:%M:%S')
data['TimeSunSet'] = pd.to_datetime(data['TimeSunSet'], format='%H:%M:%S')
data['DayLength(s)'] = data['TimeSunSet'].dt.hour*60*60 \
                           + data['TimeSunSet'].dt.minute*60 \
                           + data['TimeSunSet'].dt.second \
                           - data['TimeSunRise'].dt.hour*60*60 \
                           - data['TimeSunRise'].dt.minute*60 \
                           - data['TimeSunRise'].dt.second
data.drop(['Data','Time','TimeSunRise','TimeSunSet'], inplace=True, axis=1)

# check for null values
# print(data.isnull().sum())

results = {}

# x1 = data[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'DayOfYear', 'TimeOfDay(s)']]
# x2 = data[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'DayOfYear']]
# x3 = data[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'TimeOfDay(s)']]
# x4 = data[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'TimeOfDay(s)']]

labels = ['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'DayOfYear', 'TimeOfDay(s)']
variations = []

variations = combinations(variations,labels)

print(variations)


# for i in variations:
#     print(i)

y = data['Radiation']

# for x in variations:
#     data[x]
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
#     linReg= LinearRegression()
#     linReg.fit(x_train, y_train)
#     linReg_pred = linReg.predict(x_test)
#     results[str(r2_score(y_test, linReg_pred))] = x
#     print(r2_score(y_test, linReg_pred))


#print('R^2 score = '+str(r2_score(y_test, linReg_pred)))
#pp.pprint(results)


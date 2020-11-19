import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn import preprocessing, model_selection, neighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def convertHourToSec(time_in_hours):
    sec = time_in_hours[-2:]
    min = time_in_hours[3:5]
    hour = time_in_hours[:2]
    if sec[0] == "0":
        sec = sec[1]
    if min[0] == "0":
        min = min[1]
    if hour[0] == "0":
        hour = hour[1]
    sec = int(sec)
    min = int(min)
    hour = int(hour)

    return sec + (min*60) + (hour*60*60)


def timeAwayFromNight(sunrise, sunset, time):
    sunrise = convertHourToSec(sunrise)
    sunset = convertHourToSec(sunset)
    time = convertHourToSec(time)
    result = 0
    midpoint = (sunrise + sunset)/2
    if time <= sunrise or time >= sunset:
        result = 0
    elif time > sunrise and time <= midpoint:
        result = time - sunrise
    elif time > midpoint and time < sunset:
        result = sunset - time
    return result

data = pd.read_csv("data/SolarPrediction.csv")

# check for null values
# print(data.isnull().sum())

data['SunElevation'] = data.apply(lambda row: timeAwayFromNight(row.TimeSunRise ,row.TimeSunSet, row.Time), axis=1)
data.drop(columns = ['UNIXTime', 'Data', 'Time', 'TimeSunRise','TimeSunSet'], inplace = True)

# remove outlier
data = data.drop(6465)

# Normalize data
data['Temperature'] = (data['Temperature'] - data['Temperature'].min()) / (data['Temperature'].max() - data['Temperature'].min())
data['Pressure'] = (data['Pressure'] - data['Pressure'].min()) / (data['Pressure'].max() - data['Pressure'].min())
data['Humidity'] = (data['Humidity'] - data['Humidity'].min()) / (data['Humidity'].max() - data['Humidity'].min())
data['Speed'] = (data['Speed'] - data['Speed'].min()) / (data['Speed'].max() - data['Speed'].min())
data['SunElevation'] = (data['SunElevation'] - data['SunElevation'].min()) / (data['SunElevation'].max() - data['SunElevation'].min())

# x = np.array(data.drop(['Radiation'],1))

x1 = data[['Temperature', 'Pressure', 'Humidity', 'Speed', 'SunElevation']]
x2 = data[['Temperature', 'Pressure', 'Humidity', 'SunElevation']]
x3 = data[['Temperature', 'Humidity', 'Speed', 'Pressure']]
x4 = data[['Temperature', 'Speed', 'SunElevation', 'Humidity']]
x5 = data[['Temperature', 'SunElevation', 'Pressure', 'Humidity',]]
x6 = data[['Temperature', 'SunElevation']]
x7 = data[['Pressure', 'Humidity', 'Speed']]

variations = [x1,x2,x3,x4,x5,x6,x7]


y = np.array(data['Radiation'])

for x in variations:
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
    linReg= LinearRegression()
    linReg.fit(x_train, y_train)
    linReg_pred = linReg.predict(x_test)
    print(x.head(0))
    print(r2_score(y_test, linReg_pred))

# example = np.array([50, 30.65, 60, 311.67, 3.2, 11826])
# example = example.reshape(1,-1)
# prediction = linReg.predict(example)
# print(prediction)

# ['Temperature', 'Pressure', 'Humidity', 'Speed', 'SunElevation'] -> 0.7944892069004295
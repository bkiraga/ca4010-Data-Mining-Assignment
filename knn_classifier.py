import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn import preprocessing, model_selection, neighbors
from sklearn.neighbors import KNeighborsClassifier
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

def rankSolarRadiationtoCategories(row, min, max):
    mid = (max + min)/2
    quart = mid/2
    if row < quart:
        return 1
    elif row >= quart and row <= mid:
        return 2
    elif row > mid and row <= (quart*3):
        return 3
    else:
        return 4


data = pd.read_csv("data/SolarPrediction.csv")

# check for null values 
# print(data.isnull().sum())

# create a new column that combines time with time of sunset/sunrise
data['SunElevation'] = data.apply(lambda row: timeAwayFromNight(row.TimeSunRise ,row.TimeSunSet, row.Time), axis=1)
data.drop(columns = ['UNIXTime', 'Data', 'Time', 'TimeSunRise','TimeSunSet'], inplace = True)

# remove outlier
data = data.drop(6465)

# Normalize data
data['Temperature'] = (data['Temperature'] - data['Temperature'].min()) / (data['Temperature'].max() - data['Temperature'].min())
data['Pressure'] = (data['Pressure'] - data['Pressure'].min()) / (data['Pressure'].max() - data['Pressure'].min())
data['Humidity'] = (data['Humidity'] - data['Humidity'].min()) / (data['Humidity'].max() - data['Humidity'].min())
data['Speed'] = (data['Speed'] - data['Speed'].min()) / (data['Speed'].max() - data['Speed'].min())
data['WindDirection(Degrees)'] = (data['WindDirection(Degrees)'] - data['WindDirection(Degrees)'].min()) / (data['WindDirection(Degrees)'].max() - data['WindDirection(Degrees)'].min())
data['SunElevation'] = (data['SunElevation'] - data['SunElevation'].min()) / (data['SunElevation'].max() - data['SunElevation'].min())

max = data['Radiation'].max()
min = data['Radiation'].min()
data['Radiation'] = data.apply(lambda row: rankSolarRadiationtoCategories(row.Radiation, min, max), axis=1)


# x = np.array(data.drop(['Radiation'],1))
# y = np.array(data['Radiation'])

# x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

# classifier = neighbors.KNeighborsClassifier(n_neighbors=6)
# classifier.fit(x_train, y_train)
# accuracy = classifier.score(x_test, y_test)
# print(accuracy)

# print(data.head(10))

x1 = data[['Temperature', 'Pressure', 'Humidity', 'Speed', 'WindDirection(Degrees)', 'SunElevation']]       # Removed all of them
x2 = data[['Temperature', 'Pressure', 'Humidity', 'Speed', 'SunElevation']]                                 # Removed Wind direction
x3 = data[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'SunElevation']]                # Removed Speed
x4 = data[['Temperature', 'Pressure', 'WindDirection(Degrees)', 'SunElevation']]                            # Removed Humidity
x5 = data[['Temperature', 'Humidity', 'Speed', 'WindDirection(Degrees)', 'SunElevation']]                   # Removed Pressure
x6 = data[['Temperature', 'Pressure', 'Humidity', 'Speed', 'WindDirection(Degrees)']]                       # Removed Sun Elevation
x7 = data[['Pressure', 'Humidity', 'Speed', 'WindDirection(Degrees)', 'SunElevation']]                      # Removed Temperature
x8 = data[['Temperature', 'SunElevation']]                                                                  # Removed Everything except Temperature, SunElevation
x9 = data[['Pressure', 'Humidity', 'Speed', 'WindDirection(Degrees)']]                                      # Removed Temperature, SunElevation

variations = [x1, x2, x3, x4, x5, x6, x7, x8, x9]

y = np.array(data['Radiation'])

for x in variations:
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
    classifier = neighbors.KNeighborsClassifier(n_neighbors=6)
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test, y_test)
    print(x.columns)
    print(accuracy)
    print("")


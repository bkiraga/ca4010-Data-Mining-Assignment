from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("data/SolarPrediction.csv")

# check for null values
print(data.isnull().sum())

#time_in_hours = string representation of time eg 18:45:17 (hour:min:sec)
#it gets converted to seconds
def convertHourToSec(time_in_hours):
    sec = time_in_hours[-2:]
    min = time_in_hours[3:5]
    hour = time_in_hours[:2]
    if sec[0] == "0":
        sec = sec[1]
    if min[0] == "0":
        sec = sec[1]
    if hour[0] == "0":
        sec = sec[1]
    sec = int(sec)
    min = int(min)
    hour = int(hour)

    return sec + (min*60) + (hour*60*60)


# create a column with a value showing the amount of seconds away from sunset/sunrise

def timeAwayFromNight(sunrise, sunset, time):
    result = 0
    midpoint = (sunrise + sunset)/2
    if time <= sunrise or time >= sunset:
        result = 0
    elif time > sunrise and time <= midpoint:
        result = time - sunrise
    elif time > midpoint and time < sunset:
        result = sunset - time
    return result


del data['UNIXTime']
del data['Data']
del data['Time']
del data['TimeSunRise']
del data['TimeSunSet']


print(data.head(10))
print(data.shape)

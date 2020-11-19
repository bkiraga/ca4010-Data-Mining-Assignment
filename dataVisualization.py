import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from datetime import datetime
from pytz import timezone
import pytz


data = pd.read_csv("data/SolarPrediction.csv")
# remove outlier
data = data.drop(6465)
data = data.sort_values(['UNIXTime'], ascending = [True])

column = data["UNIXTime"]
max_value = column.max()
min_value = column.min()
print(data.loc[data['UNIXTime'] == min_value])
print(data.loc[data['UNIXTime'] == max_value])

print(data.head())

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

print(data.head())

grouped_m=data.groupby('MonthOfYear').mean().reset_index()
grouped_w=data.groupby('WeekOfYear').mean().reset_index()
grouped_d=data.groupby('DayOfYear').mean().reset_index()
grouped_h=data.groupby('TimeOfDay(h)').mean().reset_index()

f, ((axis1, axis2), (axis3, axis4), (axis5, axis6), (axis7, axis8)) = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(14,12))
axis3.set_ylim(45,60)
axis5.set_ylim(30.36,30.46)
axis7.set_ylim(60,85)

axis1.set_title('Mean Radiation by Hour')
pal = sb.color_palette("Spectral", len(grouped_h))
place = grouped_h['Radiation'].argsort().argsort() 
g = sb.barplot(x="TimeOfDay(h)", y='Radiation', data=grouped_h, palette=np.array(pal[::-1])[place], ax=axis1)
axis1.set_xlabel('')

axis2.set_title('Mean Radiation by Month')
pal = sb.color_palette("Spectral", len(grouped_m))
place = grouped_m['Radiation'].argsort().argsort() 
g = sb.barplot(x="MonthOfYear", y='Radiation', data=grouped_m, palette=np.array(pal[::-1])[place], ax=axis2)
axis2.set_xlabel('')

axis3.set_title('Mean Temperature by Hour')
pal = sb.color_palette("Spectral", len(grouped_h))
place = grouped_h['Temperature'].argsort().argsort() 
g = sb.barplot(x="TimeOfDay(h)", y='Temperature', data=grouped_h, palette=np.array(pal[::-1])[place], ax=axis3)
axis3.set_xlabel('')

axis4.set_title('Mean Temperature by Month')
pal = sb.color_palette("Spectral", len(grouped_m))
place = grouped_m['Temperature'].argsort().argsort() 
g = sb.barplot(x="MonthOfYear", y='Temperature', data=grouped_m, palette=np.array(pal[::-1])[place], ax=axis4)
axis4.set_xlabel('')

axis5.set_title('Mean Pressure by Hour')
pal = sb.color_palette("Spectral", len(grouped_h))
place = grouped_h['Pressure'].argsort().argsort() 
g = sb.barplot(x="TimeOfDay(h)", y='Pressure', data=grouped_h, palette=np.array(pal[::-1])[place], ax=axis5)
axis5.set_xlabel('')

axis6.set_title('Mean Pressure by Month')
pal = sb.color_palette("Spectral", len(grouped_m))
place = grouped_m['Pressure'].argsort().argsort() 
g = sb.barplot(x="MonthOfYear", y='Pressure', data=grouped_m, palette=np.array(pal[::-1])[place], ax=axis6)
axis6.set_xlabel('')

axis7.set_title('Mean Humidity by Hour')
pal = sb.color_palette("Spectral", len(grouped_h))
place = grouped_h['Humidity'].argsort().argsort() 
g = sb.barplot(x="TimeOfDay(h)", y='Humidity', data=grouped_h, palette=np.array(pal[::-1])[place], ax=axis7)

axis8.set_title('Mean Humidity by Month')
pal = sb.color_palette("Spectral", len(grouped_m))
place = grouped_m['Humidity'].argsort().argsort() 
g = sb.barplot(x="MonthOfYear", y='Humidity', data=grouped_m, palette=np.array(pal[::-1])[place], ax=axis8)

plt.show()

trainData = data.drop(['TimeOfDay(h)', 'TimeOfDay(m)', 'TimeOfDay(s)', 'UNIXTime', 'MonthOfYear', 'WeekOfYear'], inplace=False, axis=1)

plt.figure(figsize=(15,15))
sb.heatmap(trainData.corr(),annot=True,cmap='Spectral')
plt.show()
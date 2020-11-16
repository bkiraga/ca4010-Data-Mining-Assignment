from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("data/SolarPrediction.csv")
print(dataset.head(20))
print(dataset.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

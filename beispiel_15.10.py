from matplotlib.pyplot import sca
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['Kelchblatt-laenge', 'Kelchblatt-breite', 'Blumenblatt-laenge', 'Blumenblatt-breite', 'Klasse']
dataset = pd.read_csv(url, names=names)
#print(dataset.head())
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# print(x)
# print(y)

# Überanspassung vermeiden: Datensatz teilen in Training/ Testdaten

from sklearn.model_selection import train_test_split
# 80%/ 20% Training/Test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)
y_vorhersage = classifier.predict(x_test)

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_vorhersage))
print(classification_report(y_test, y_vorhersage))
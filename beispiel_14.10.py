import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x = np.array([5,10,15,17,22,27]).reshape(-1,1)
y = np.array([5,20,13,22,44,60])
model = LinearRegression().fit(x,y)
r_quadrat = model.score(x,y)
print("Bestimmtheitsmaß:", r_quadrat)
print("Schnittpunkt (beta_0)", model.intercept_)
print("Steigung (beta_1",model.coef_)
y_vorhersage = model.predict(x)
print("Y-Vorhersagen:", y_vorhersage)
#_vorhersage= model.intercept_ + model.coef_ *x

plt.plot(x,y_vorhersage)
plt.show()
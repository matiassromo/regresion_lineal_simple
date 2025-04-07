# Regresión Lineal Simple: Previous Scores vs Performance Index

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar el dataset
dataset = pd.read_csv(r'C:\Users\Usuario\Desktop\PC\UDLA\SEMESTRE 8\INTELIGENCIA ARTIFICIAL I\regresion_lineal_simple\Student_Performance.csv')

# Variable independiente: 'Previous Scores'
# Variable dependiente: 'Performance Index'
X = dataset[['Previous Scores']].values
y = dataset['Performance Index'].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Crear y entrenar el modelo
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predecir resultados
y_pred = regressor.predict(X_test)

# Evaluar el modelo
from sklearn.metrics import mean_squared_error, r2_score
print("Coeficiente (pendiente):", regressor.coef_[0])
print("Intercepción:", regressor.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² score:", r2_score(y_test, y_pred))

# Gráfico del conjunto de entrenamiento
plt.scatter(X_train, y_train, color='blue', label='Datos reales')
plt.plot(X_train, regressor.predict(X_train), color='red', label='Línea de regresión')
plt.title('Previous Scores vs Performance Index (Entrenamiento)')
plt.xlabel('Previous Scores')
plt.ylabel('Performance Index')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico del conjunto de prueba
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_train, regressor.predict(X_train), color='red', label='Línea de regresión')
plt.title('Previous Scores vs Performance Index (Prueba)')
plt.xlabel('Previous Scores')
plt.ylabel('Performance Index')
plt.legend()
plt.grid(True)
plt.show()

# Regresión Lineal Simple

#Explicaciones clave:
#Importación de datos: Se usa el archivo insurance.csv con columnas 'age' y 'charges'.
#División del dataset: train_test_split divide en entrenamiento (2/3) y prueba (1/3).
#Entrenamiento: LinearRegression de sklearn ajusta un modelo a los datos.
#Visualización: Se grafican los datos reales y la línea de regresión para ambos conjuntos.

'''Interpretación: Este modelo usa regresión lineal simple para predecir los costos médicos en función de la edad.
Se observa una relación creciente (línea azul), pero no todos los puntos se alinean perfectamente.
Esto indica que la edad influye, pero no explica todo, ya que otros factores como si la persona fuma, su IMC, y enfermedades previas también afectan los costos.'''

# Importar bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
# Reemplazamos Salary_Data.csv por insurance.csv y tomamos 'age' y 'charges'
dataset = pd.read_csv(r'C:\Users\Usuario\Desktop\PC\UDLA\SEMESTRE 8\INTELIGENCIA ARTIFICIAL I\regresion_lineal_simple\insurance.csv')

# Usamos iloc para mantener la estructura del profe:
# Supongamos que 'age' es la columna 0 y 'charges' es la columna 6
X = dataset.iloc[:, [0]].values   # 'age'
y = dataset.iloc[:, 6].values     # 'charges'

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Entrenar el modelo de Regresión Lineal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predecir los resultados del conjunto de prueba
y_pred = regressor.predict(X_test)

# Visualizar los resultados del conjunto de entrenamiento
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Edad vs Costos Médicos (Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Costos Médicos (USD)')
plt.show()

# Visualizar los resultados del conjunto de prueba
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Edad vs Costos Médicos (Prueba)')
plt.xlabel('Edad')
plt.ylabel('Costos Médicos (USD)')
plt.show()

# Regresión Lineal Simple usando Student_Performance.csv

# Explicaciones clave:
# Importación de datos: El archivo Student_Performance.csv contiene columnas como 'Hours Studied' y 'Performance Index'.
# División del dataset: Usamos train_test_split para dividir los datos en un conjunto de entrenamiento (2/3) y prueba (1/3).
# Entrenamiento del modelo: Usamos LinearRegression de sklearn.
# Visualización de resultados: Se grafican los datos reales y la línea de regresión.

# Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv(r'C:\Users\Usuario\Desktop\PC\UDLA\SEMESTRE 8\INTELIGENCIA ARTIFICIAL I\regresion_lineal_simple\Student_Performance.csv')

# Variable independiente: 'Hours Studied'
# Variable dependiente: 'Performance Index'
X = dataset[['Hours Studied']].values
y = dataset['Performance Index'].values

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
plt.scatter(X_train, y_train, color='red', label='Datos reales')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Línea de regresión')
plt.title('Horas de Estudio vs Índice de Rendimiento (Entrenamiento)')
plt.xlabel('Horas de Estudio')
plt.ylabel('Índice de Rendimiento')
plt.legend()
plt.grid(True)
plt.show()

# Visualizar los resultados del conjunto de prueba
plt.scatter(X_test, y_test, color='red', label='Datos reales')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Línea de regresión')
plt.title('Horas de Estudio vs Índice de Rendimiento (Prueba)')
plt.xlabel('Horas de Estudio')
plt.ylabel('Índice de Rendimiento')
plt.legend()
plt.grid(True)
plt.show()

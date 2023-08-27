# Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)
# Emiliano Vásquez Olea - A01707035

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
  Método de One Hot Encoding para asignar las categorías de la columna "Species" a valores numéricos.
  
  Esto crea nuevos atributos en la tabla de entradas o x, correspondiendo a las diferentes especies existentes,
  y se asigna el valor 0 o 1 dependiendo si el registro del pez corresponde o no a esa especie. Al crear estos
  atributos se incrementa el número de parámetros en el modelo.
'''
def species_one_hot_encoding(df):
  encoded_species_df = pd.get_dummies(df['Species'], prefix='Species')
  df = pd.concat([df, encoded_species_df], axis = 1).drop(['Species'], axis = 1) # Se elimina la columna inicial de especies

  return df

'''
  Estandarización de datos en el dataframe utilizando la media y desviación estándar de la columna.
  No se aplica esto a las columnas agregadas por el One Hot Encoding, ya que se encuentran entre 0 y 1 únicamente.
'''
def normalize_df(df):
  normalized_df = df.copy()
  normalized_df[[
      'Length1',
      'Length2',
      'Length3',
      'Height',
      'Width']] = normalized_df[[
          'Length1',
          'Length2',
          'Length3',
          'Height',
          'Width']].apply(lambda x: (x - x.mean()) / x.std())

  normalized_df = normalized_df.fillna(0)

  return normalized_df

'''
  Función para evaluar el modelo actual en cada una de las instancias de un dataframe.
  Regresa una serie con los resultados de cada evaluación
'''
def evaluate_h(params, df_x):
  # Se multiplica la fila del dataframe por los parametros, excluyendo el último que representa la constante b (bias)
  results = df_x.multiply(params[:-1], axis = 1)
  results = results.sum(axis = 1) + params[-1] # Agregamos el valor de b a los resultados
  results.name = 'Predicted weight'

  return results

'''
  Se calcula la suma de los errores entre las predicciones del modelo y los valores de Y reales para ser
  usados en el descenso de gradiente.
'''
def get_error_sum(params, param_index, df_x, df_y):
  h_results = evaluate_h(params, df_x) # Resultados del modelo
  errors = h_results.sub(df_y.squeeze()) # Error

  if param_index != len(params)-1: # Se multiplica el error por el valor de x correspondiente del parámetro
    errors = errors.multiply(df_x.iloc[:, param_index])

  return errors.sum()

'''
  Esta función se encarga de recalcular un parámetro a partir de la pendiente en nuestra función del costo,
  el parámetro anterior y nuestro learning rate.
'''
def get_updated_param(params, param_index, df_x, df_y, learning_rate):
  error_sum = get_error_sum(params, param_index, df_x, df_y)

  return params[param_index] - (learning_rate * (1 / len(df_x)) * error_sum)

'''
  Ejecución del descenso de gradiente sobre todos los parámetros del modelo. Se actualiza el modelo.
'''
def gradient_descent(params, df_x, df_y, learning_rate):
  for param_index, _ in enumerate(params):
    params[param_index] = get_updated_param(params, param_index, df_x, df_y, learning_rate)

  return params

'''
  Esta función se encarga de evaluar el modelo previamente entrenado sobre otro conjunto de datos que conforma
  el test. Posteriormente se muestra el error junto con una grafica comparando los valores reales con los
  predecidos por el modelo.

  El one hot encoding y estandarización de los datos debe ser aplicado previamente.
'''
def show_test_results(params, df_x_test, df_y_test):
  # Se calculan los errores similar a get_error_sum, pero obteniendo los porcentajes
  h_results = evaluate_h(params, df_x_test)
  percent_errors = h_results.sub(df_y_test.squeeze()).div(df_y_test.squeeze())

  RMPSE = np.sqrt(np.square(percent_errors).mean())

  # Se agrupan los resultados en un dataframe, ordenando los elementos por el peso real
  df_test_results = pd.concat([df_y_test.squeeze(), h_results], axis = 1)
  df_test_results = df_test_results.sort_values(by = ['Weight']).reset_index(drop = True)

  print('Porcentaje de error:', RMPSE)

  plt.scatter(
      x = df_test_results.index,
      y = df_test_results['Weight'],
      label = 'Peso real')

  plt.scatter(
      x = df_test_results.index,
      y = df_test_results['Predicted weight'],
      label = 'Predicción')
  
  plt.title('Predicciones y valores reales de peso')
  plt.legend(loc = 'upper left')
  
  plt.show() # Mostrar gráfico

'''
  Función principal para el entrenamiento del modelo de regresión lineal. Aquí se actualizan los parámetros a 
  partir del número de épocas asignadas usando batches de los datos de entrenamiento.
'''
def linear_regression(df_x_train, df_y_train, epochs, batch_size, alpha):
  params = np.ones((1, df_x_train.shape[1] + 1))[0] # Arreglo con los coeficientes iniciales

  for _ in range(epochs):
    # Se crean dos listas separando los batches de X y Y
    x_train_bathes = [df_x_train_normalized[i:i+batch_size] for i in range(0, len(df_x_train_normalized), batch_size)]
    y_train_bathes = [df_y_train[i:i+batch_size] for i in range(0, len(df_y_train), batch_size)]

    # Se ejecuta el descenso de gradiente sobre las listas con los batches, actualizando los coeficientes para cada uno.
    for batch_index, _ in enumerate(x_train_bathes):
      params = gradient_descent(params, x_train_bathes[batch_index], y_train_bathes[batch_index], alpha)

  return params

df = pd.read_csv('Fish.csv') # Carga de datos

df.drop(df[df['Weight'] < 1].index, inplace = True) # Limpiar instancia con valor faltante de peso

df = species_one_hot_encoding(df) # Se aplica el one hot encoding

# Se divide el dataset en datos de entrenamiento (80%) y de prueba (20%)
df_test = df.sample(frac = 0.2, random_state = 20)
df_train = df.drop(df_test.index)

df_x_train = df_train.drop(['Weight'], axis = 1)
df_y_train = df_train[['Weight']]

df_x_test = df_test.drop(['Weight'], axis = 1)
df_y_test = df_test[['Weight']]

# Definición de hiperparámetros: learning rate, iteraciones y tamaño de los batches
alpha = 0.03
epochs = 350
batch_size = 31

# Se aplica la estandarización de los datos
df_x_train_normalized = normalize_df(df_x_train)
df_x_test_normalized = normalize_df(df_x_test)

params = linear_regression(df_x_train_normalized, df_y_train, epochs, batch_size, alpha) # Entrenamiento

print('Parámetros: ', params)

show_test_results(params, df_x_test_normalized, df_y_test) # Pruebas
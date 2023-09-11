# Análisis y Reporte sobre el desempeño del modelo. (Portafolio Análisis)
# Emiliano Vásquez Olea - A01707035

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

'''
  Esta función es utilizada de forma separada para probar diferentes valores de un hiperparámetro
  en el modelo de Random Forest. Los puntajes de precisión al utilizar el modelo con estos
  distintos valores es graficado, comparando los datos de entrenamiento con el conjunto de validación.
  En este caso, se ve la implementación para probar diferentes valores de n_estimators.
'''
def test_model_params(df_x_train, df_y_train, df_x_val, df_y_val):
  estimator_opts = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220] # Opciones para el hiperparametro
  train_results = []
  val_results = []

  for trees in estimator_opts:
    model = RandomForestClassifier(n_estimators = trees, n_jobs = -1, max_features = None, random_state = 50)
    model.fit(df_x_train, df_y_train)

    train_results.append(accuracy_score(df_y_train, model.predict(df_x_train)))
    val_results.append(accuracy_score(df_y_val, model.predict(df_x_val)))

  plt.plot(estimator_opts, train_results, label = "Puntaje en entrenamiento")
  plt.plot(estimator_opts, val_results, color = "orange", label = "Puntaje de validación")

  plt.legend()
  plt.ylabel('Accuracy Score')
  plt.xlabel('n_estimators')

  plt.show()

'''
  Estandarización de datos en el dataframe utilizando el scalar Min Max de scikit-learn.
  Aplicamos este escalado a todos nuestros datos y regresamos los datos contenidos en un DataFrame.
'''
def scale_df(df):
  scaler = MinMaxScaler()
  data_scaled = scaler.fit_transform(df)
  df_scaled = pd.DataFrame(data_scaled)

  return df_scaled

'''
  Esta función se encarga de generar y entrenar el modelo de clasificación de Random Forest.
  Las entradas para esta función son únicamente los dataframes de las vaiables x y la variable y,
  los hiperparámetros para el modelo son definidos internamente para esta implementación.
'''
def train_forest_classifier(df_x_train, df_y_train):
  model = RandomForestClassifier(n_estimators = 200, n_jobs = -1, max_features = None, random_state = 50)
  model.fit(df_x_train, df_y_train)

  return model

'''
  Esta función permite evaluar los datos a partir de un dataset de entrada y graficar los resultados usando un 
  scatter plot. Aquí se utilizan los atributos de RAM y Altura en pixeles como ejemplo.
'''
def show_scatter_results(model, df_x_test_scaled, df_x_test):
  y_pred = model.predict(df_x_test_scaled)

  plt.scatter(df_x_test['ram'], df_x_test['px_height'], c = y_pred) # Tomamos dos atributos de ejemplo para el grafico

  plt.title("Clasificación")
  plt.xlabel("RAM")
  plt.ylabel("Pixel Height")

  plt.show()

'''
  Esta función permite probar el desempeño del modelo de clasificación con los valores separados
  de "test". Primero se generan las predicciones, se comparan con los valores reales y se
  calcula el puntaje/precisión del modelo. Finalmente, se despliega esta información en la consola.
'''
def show_test_results(model, df_x_test, df_y_test):
  y_pred = model.predict(df_x_test)
  print('Random forest accuracy score:', accuracy_score(df_y_test, y_pred))
  print('Predictions:')

  # Integramos la información en un dataframe
  df_results = pd.DataFrame({'Real price range':df_y_test['price_range'],
                             'Predicted price range':y_pred
                            }).reset_index(drop = True)
  print(df_results)

# El dataset cuenta con un archivo llamado test.csv, pero no contiene las etiquetas reales
# por lo que no podemos comparar estos resultados
df = pd.read_csv('train.csv') # Carga de datos

# Se divide el dataset en datos de entrenamiento (85%) y de prueba (15%)
df_test = df.sample(frac = 0.15, random_state = 50)
df_train = df.drop(df_test.index)

# Dividimos del dataset de entrenamiento un conjunto para validación (15% de train)
df_val = df_train.sample(frac = 0.15, random_state = 50)
df_train = df_train.drop(df_val.index)

df_x_train = df_train.drop(['price_range'], axis = 1)
df_y_train = df_train[['price_range']]

df_x_test = df_test.drop(['price_range'], axis = 1)
df_y_test = df_test[['price_range']]

df_x_val = df_val.drop(['price_range'], axis = 1)
df_y_val = df_val[['price_range']]

# Se aplica el escalado sobre los datos
df_x_train_scaled = scale_df(df_x_train)
df_x_test_scaled = scale_df(df_x_test)
df_x_val_scaled = scale_df(df_x_val)

#test_model_params(df_x_train_scaled, df_y_train, df_x_val_scaled, df_y_val)

model = train_forest_classifier(df_x_train_scaled, df_y_train) # Entrenamiento

show_test_results(model, df_x_train_scaled, df_y_train) # Pruebas con "train"
#show_scatter_results(model, df_x_train_scaled, df_x_train)

show_test_results(model, df_x_test_scaled, df_y_test) # Pruebas con "test"
#show_scatter_results(model, df_x_test_scaled, df_x_test)

show_test_results(model, df_x_val_scaled, df_y_val) # Pruebas con "validation"
#show_scatter_results(model, df_x_val_scaled, df_x_val)

input('Presiona ENTER para cerrar ') # Agregamos un input para prevenir que la ventana de la consola se cierre
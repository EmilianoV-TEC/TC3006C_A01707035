# Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. (Portafolio Implementación)
# Emiliano Vásquez Olea - A01707035

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

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
  model = RandomForestClassifier(n_estimators = 150, n_jobs = -1, max_features = None, random_state = 50)
  model.fit(df_x_train, df_y_train)

  return model

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

df_x_train = df_train.drop(['price_range'], axis = 1)
df_y_train = df_train[['price_range']]

df_x_test = df_test.drop(['price_range'], axis = 1)
df_y_test = df_test[['price_range']]

# Se aplica el escalado sobre los datos
df_x_train_scaled = scale_df(df_x_train)
df_x_test_scaled = scale_df(df_x_test)

model = train_forest_classifier(df_x_train_scaled, df_y_train) # Entrenamiento

show_test_results(model, df_x_test_scaled, df_y_test) # Pruebas

input('Presiona ENTER para cerrar ') # Agregamos un input para prevenir que la ventana de la consola se cierre
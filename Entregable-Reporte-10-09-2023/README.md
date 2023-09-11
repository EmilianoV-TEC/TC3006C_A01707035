# Momento de Retroalimentación: Módulo 2 Análisis y Reporte sobre el desempeño del modelo. (Portafolio Análisis)
### Emiliano Vásquez Olea - A01707035

Archivos: Entrega_FrameworkyReporte_A01707035.py, train.csv, test.csv, Reporte_M2_A01707035.pdf

Requisitos: Una versión actualizada de **Python 3** junto con las librerías **numpy**, **matplotlib**, **pandas** y **scikit-learn**

La ejecución del código puede tardar algunos segundos.

En el documento Reporte_M2_A01707035.pdf se encuentra el reporte ligado a esta entrega, donde se explican factores como el proceso de análisis de datos, entrenamiento y ajuste de parámetros.

## Dataset
Para esta entrega se utiliza el dataset Mobile Price Classification (https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification), que contiene información y especificaciones sobre distintos teléfonos celulares.

Este dataset ya se encuentra dividido en un set de entrenamiento y uno de prueba, que contienen las siguientes columnas o atributos:

- battery_power: La cantidad de energía, medida en mAh, que puede almacenar el dispositivo
- blue: Si el teléfono cuenta con Bluetooth
- clock_speed: Velocidad del microprocesador
- dual_sim: Si el dispositivo soporta Dual SIM
- fc: Megapixeles de la cámara frontal
- four_g: Si el dispositivo soporta 4G
- int_memory: Memoria interna en GB
- m_dep: Profundidad del dispositivo en cm
- mobile_wt: Peso del dispositivo
- n_cores: Número de núcleos del procesador
- pc: Megapixeles de la cámara principal
- px_height: Altura de la resolución en pixeles
- px_width: Ancho de la resolución en pixeles
- ram: Memoria RAM del dispositivo en MB
- sc_h: Altura de la pantalla en cm
- sc_w: Ancho de la pantalla en cm
- talk_time: Tiempo que dura una sola carga de la batería en llamada telefónica
- three_g: Si el dispositivo soporta 3G
- touch_screen: Si el dispositivo tiene pantalla táctil
- wifi: Si el dispositivo tiene acceso a WiFi
- price_range: El rango de precio al que pertenece el teléfono (bajo, medio, alto o muy alto)

En este caso, la variable a predecir (y) es el rango de precios al que pertenece el dispositivo. El objetivo de este modelo es estimar el valor que se le debe atribuir a un teléfono de acuerdo a lo visto en el mercado.

Es importante mencionar que los datos en el archivo test.csv no cuentan con la columna price_range, que es la variable que queremos predecir, por lo que de igual forma se hará una separación de los datos extraídos de test.csv.

## Técnica implementada

Aunque a simple vista de la descripción del dataset uno asumiría que se debe trabajar con una regresión, debido a que se busca predecir un valor del precio. Sin embargo, el atributo que queremos predecir es realmente una clase, el rango de precios, que está separado en 4 categorías. Es por esto que se debe trabajar con un modelo de clasificación, que en este caso decidí utilizar Random Forest

Random Forest es considerado un modelo de Ensemble, que se refiere a aquellos algoritmos que utilizan un conjunto de modelos más pequeños y los juntan de cierta forma para obtener las predicciones deseadas. Random Forest utiliza árboles de decisión con el método de bagging, indicando que estos modelos son entrenados de forma paralela. Para la implementación de este modelo se utiliza la librería scikit-learn con la clase RandomForestClassifier.

Al ejecutar el código se lleva a cabo el entrenamiento del modelo y posteriormente se utiliza con los datos de prueba, mostrando el puntaje de precisión calculado con la misma librería de scikit-learn, así como las predicciones junto con los rangos de precio reales.

## Parámetros finales y notas

Los parámetros del modelo son definidos al momento de crear el RandomForestClassifier en la línea 55 del código, entre los que fueron asignados se encuentran:

- n_estimators: 200, indica el número de árboles de decisión utilizados en el bósque
- n_jobs: -1, El número de trabajos u operaciones que corren en paralelo, asignar un -1 indica que se utilicen todos los procesadores (para su uso en notebooks o servicios que cuentan con acceso a varios procesadores)
- max_features: None, indica la cantidad máxima de features que se toman en cuenta al definir los nodos en los arboles de decisión.
- random_state: 50, una semilla utilizada para realizar diferentes operaciones dentro del modelo.

Este modelo cuenta con más parámetros que pueden ser configurados, sin embargo, estos se mantenieron con sus valores por defecto, por ejemplo:

- max_depth: None, la profundidad máxima de un arbol
- Bootstrap: True, si se utiliza el método de Bootstrapping para obtener fragmentos del dataset.

Estos parámetros se asignaron después de pruebas con distintos valores para el entrenamiento y el análisis del rendimiento del modelo, que se encuentra documentado en el reporte de esta entrega.


# Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)
### Emiliano Vásquez Olea - A01707035

Archivos: Entrega_M2_A01707035.py, Fish.csv

Requisitos: Una versión actualizada de **Python 3** junto con las librerías **numpy**, **pandas** y **matplotlib**

La ejecución del código tarda algunos segundos.

## Dataset
Para esta entrega se utiliza el dataset Fish Market (https://www.kaggle.com/datasets/aungpyaeap/fish-market), el cual contiene registros sobre diferentes especies de peces, principalmente en cuanto a sus dimensiones.

Se cuenta con datos de 7 diferentes especies con las siguientes columnas o atributos:

- Species: El nombre de la especie del pescado
- Weight: El peso del pescado en gramos
- Length1: Longitud vertical en centímetros
- Length2: Longitud diagonal en centímetros
- Length3: Longitud transversal en centímetros
- Height: Altura del pescado en centímetros
- Width: Anchura diagonal en centímetros

Entre estas características seleccioné el peso como la variable a predecir (y), al ser un aspecto importante dentro del contexto de una pescadería.

## Técnica implementada

Para las predicciones del peso se emplea un modelo de regresión lineal, ya que nos permite calcular valores numéricos en relación a los datos de entrada. Para el entrenamiento de este modelo de aprendizaje automático se utiliza el descenso de gradiente por lotes o batches, minimizando el error dado por la función MSE (error cuadrado medio).

Al ejecutar el código se lleva a cabo el entrenamiento del modelo y posteriormente se utiliza con los datos de prueba, mostrando los coeficientes finales del modelo, el error en porcentaje (RMSPE) y una gráfica con las predicciones y valores reales de peso.

Una alternativa para trabajar con este dataset es entrenar un modelo logístico para clasificar la especie de un pescado a partir de sus otros atributos.

## Parámetros y notas

Los hiperparámetros se encuentran definidos en el código en las líneas 156, 157 y 158:

- Alfa o learning rate: 0.03
- Épocas: 350
- Tamaño de lote: 31

Estos parámetros se asignaron después de pruebas con distintos valores para el entrenamiento, obteniendo un buen resultado sobre los datos de prueba con este conjunto. El tamaño del lote de 31 fué asignado ya que divide en 5 el dataset de entrenamiento con 155 registros.

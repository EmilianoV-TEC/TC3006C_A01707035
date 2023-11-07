# Momento de Retroalimentación: Módulo 2 Implementación de un modelo de deep learning. (Portafolio Implementación)
### Emiliano Vásquez Olea - A01707035

Archivos: Entrenamiento_DL_A01707035.ipynb, Despliegue_DL_A01707035.ipyn, MobileNetV3_card_classifier.h5, Reporte_DL_A01707035.pdf

Requisitos: Una versión actualizada de **Python 3** junto con las librerías **numpy**, **matplotlib**, **pandas**, **seaborn** y **tensorflow 2 (con keras)**

La ejecución del código de entrenamiento tarda varios minutos, mientras que el despliegue unos cuantos segundos (Dependiendo del equipo de cómputo utilizado).

En el documento Reporte_DL_A01707035.pdf se encuentra el reporte ligado a esta entrega, donde se explican factores como el proceso de análisis de datos, entrenamiento y ajuste de parámetros.

## Dataset
Para esta entrega se utiliza el dataset Cards Image Dataset-Classification (https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification), que está conformado por alrededor de 8000 imágenes de cartas de una baraja, incluyendo el Joker (53 tipos de cartas). El dataset se encuentra cargado en la carpeta *card-data*.

Este dataset ya ha sido dividido en un set de entrenamiento, uno de validación y uno de prueba, cada uno en una carpeta distinta. Dentro de cada una de estas carpetas se encuentra un nuevo conjunto de 53 directorios, separando cada una de las clases o tipos de cartas con sus respectivas imágenes. Cada imágen tiene las dimensiones de 224 x 224 x 3, y son almacenadas en formato JPG, facilitando el pre-procesado de los datos.

Para este modelo de clasificación el objetivo es determinar la clase a la cual corresponde la imágen de una carta, es decir, a partir de una foto, determinar que carta es (por ejemplo, 5 de tréboles o 10 de diamantes).

## Técnica implementada

Al ser un problema de clasificación de imágenes, podemos aplicar la técnica de aprendizaje de máquina de Redes Neuronales Convolucionales. Este tipo de redes neuronales está especializada en el manejo de datos que pueden ser representados como una matriz, por ejemplo, imágenes. La diferencia principal entre las redes convolucionales y redes neuronales generales es, como su nombre lo indica, el uso de capas convolutivas, que permiten extraer diferentes características de una imagen generando nuevas matrices abstractas para el modelo. Estas capas de igual forma siguen un proceso de aprendizaje junto con el resto de la red y se suman al resto de capas densas o con otro tipo de operaciones para crear la red neuronal completa, finalizando con una capa de salida.

Para crear la red neuronal inicial se utiliza como base la arquitectura de MobileNet V3, una red convolucional relativamente ligera debido a su enfoque para el procesamiento dentro de teléfonos celulares. Integrando este modelo previamente utilizado y probado en la clasificación de otros tipos de imágenes, es posible contar con una base robusta con diferentes operaciones y capacidad de extracción de información de las imágenes, sin embargo, para completar la red neuronal y realizar predicciones, es necesario agregar algunas capas finales al modelo. Después de la base de MobileNet se agrega una capa de Global Average Pooling y dos capas densas, una con 128 neuronas y la función de activación ReLU, y la última teniendo el output final de 53 clases.

El notebook de entrenamiento, como su nombre lo indica, lleva a cabo el ajuste de parámetros o entrenamiento del modelo con la arquitectura definida, utilizando el conjunto de datos de validación como otra medida de precisión. En la carpeta de *checkpoints* se almacenan los pesos generados en el entrenamiento para cada época donde se vió una mejora en la precisión, mientras que el modelo final se almacena en la carpeta *models* al finalizar la ejecución. Por último, el archivo de despliegue permite cargar el modelo generado y probarlo con una serie de muestras extraídas de la carpeta *test* dentro del dataset.

## Parámetros iniciales y notas

Entre los parámetros iniciales del modelo, adicionales a la arquitectura utilizada, se encuentran los siguientes:

- optimizer: "adam", el optimizador utilizado en el entrenamiento del modelo, definido en el método compile
- loss: 'categorical_crossentropy', la función de pérdida utilizada en el entrenamiento
- metrics: 'acc, métricas adicionales que son observadas en el entrenamiento
- epochs: 15, cantidad de épocas o iteraciones en el entrenamiento
- steps_per_epoch: 200, pasos de ajuste realizados por época.

Como fué mencionado anteriormente, se utilizan los conjuntos de entrenamiento y validación para el proceso de ajuste, además de que se integra un callback para guardar los pesos en forma de checkpoints a lo largo del proceso. La imagen "image1_joker.jpg" es utilizada para realizar una predicción separada en el despliegue, como es mencionado en el *notebook*, a este tipo de archivos se les debe aplicar una división adicional.

Es posible que algunos de estos valores, así como la arquitectura, sean ajustados como parte del análisis del rendimiendo del modelo, que se encuentra documentado en el reporte de esta entrega.


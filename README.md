##  <center> Práctica Perceptron Multicapa </center>
### Autor: Raúl Martín Torrabadella Mendoza
### Introducción

En esta práctica abordaremos uno de los problemas fundacionales de la visión artificial: el reconocimiento de caracteres (OCR) mediante un Perceptrón Multicapa (MLP).

El objetivo no es solo que la red "funcione", sino que comprendáis la importancia de la arquitectura y los hiperparámetros en el resultado final. En el entorno profesional, la diferencia entre un modelo mediocre y uno productivo reside en la capacidad del ingeniero para ajustar estos valores sistemáticamente.

### Objetivos
Con la realización de esta práctica se persiguen los siguientes objetivos formativos:

- Implementar un flujo de trabajo de Deep Learning: Desde la ingesta de datos hasta la predicción final.

- Manejo de Librerías vs. Algoritmia Pura: Ser capaz de utilizar frameworks de alto nivel o, alternativamente, demostrar una comprensión profunda implementando el algoritmo desde cero.

- Optimización Sistemática: Aplicar técnicas de Grid Search para evitar el ajuste manual y aleatorio de hiperparámetros.

- Evaluación Crítica: Interpretar métricas más allá del accuracy global, analizando errores específicos mediante matrices de confusión.

- Documentación Profesional: Sintetizar decisiones técnicas y resultados en una memoria formal.

### Pasos

#### Paso 1: Selección y Preparación del Dataset

En mi caso hice la opción b:
Opción B (Reto): EMNIST (Letras manuscritas) o combinaciones similares.

Como se puede ver en el documento [funcionesYdatos.py](https://github.com/RMTorrabadella04/MIA_Perceptron-Multicapa/blob/master/funcionesYdatos.py).

#### Paso 2: Estrategia de Implementación

En mi caso hice la opción b:
Ruta B (Implementación Propia - "From Scratch"): Programar el MLP usando solo NumPy (implementando la propagación hacia adelante y el backpropagation manualmente). Esta opción demuestra un dominio superior de la materia y será valorada muy positivamente.

Como se puede ver en el documento [main.py](https://github.com/RMTorrabadella04/MIA_Perceptron-Multicapa/blob/master/main.py)

#### Paso 3: Diseño del Experimento (Grid Search)

No basta con entrenar una vez. Debéis diseñar una búsqueda de la arquitectura óptima.

Definid un diccionario de hiperparámetros a explorar. Ejemplos:

  - Capas ocultas: [(50,), (100,), (50, 50), (100, 50)]

  - Funciones de activación: ['relu', 'tanh', 'logistic']

  - Tasa de aprendizaje (Learning Rate): [0.001, 0.01]

Ejecutad la búsqueda (usando GridSearchCV o bucles propios si habéis hecho la implementación manual) y almacenad los resultados.

#### Paso 4: Entrenamiento y Validación del Modelo Final

Seleccionad la mejor configuración ganadora del paso anterior.

Re-entrenad el modelo con el conjunto de Train completo.

Realizad las predicciones sobre el conjunto de Test.

#### Paso 5: Análisis de Resultados

Calculad la precisión final (Accuracy).

Generad y visualizad la Matriz de Confusión.

Identificad visualmente 3 o 4 casos donde la red haya fallado (ej. confundir un 9 con un 4) y añadidlos a la memoria.

### Resultados Finales

- Accuracy:

- Matriz de Confusión:

  ![ImagenMatriz](https://github.com/RMTorrabadella04/MIA_Perceptron-Multicapa/blob/master/matriz_confusion_emnist.png)


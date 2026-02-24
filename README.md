## `<center>` Práctica Perceptron Multicapa `</center>`

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

### Estructura y Archivos

MIA_PerceotronMulticapa/

├── .gitignore

├── .python-version

├── README.md

├── funcionesYdatos.py

├── main.py

├── matriz_confusion_emnist.png

├── pyproject.toml

└── requirement.txt

#### funcionesYdatos.py

Este archivo contiene la recogida de datos, además de las funciones de activación y sus funciones derivadas.

#### main.py

Este archivo contiene toda la lógica detras del entrenamiento (Es el archivo a ejecutar).

#### matriz_confusion_emnist.png

Es la matriz de confusión de la combinación con un mayor accuracy.

#### requirement.txt

Este archivo contiene todas las librerias necesarias a instalar, para usarlo deberas ejecutar el siguiente comando, además de tener instalado python.

```
pip install -r requirement.txt
```

Cabe aclarar que si quieres usar un entorno virtual sera el siguiente proceso.

```
uv venv [NOMBRE_ENTORNO_VIRTUAL]

.[NOMBRE_ENTORNO_VIRTUAL]\Scripts\activate

# Por defecto el venv, viene sin nada instalado, por lo que instalaremos pip
python -m ensurepip

python -m pip install -r requirement.txt
```

### Resultados Finales

Los resultados finales de **mi último entrenamiento y predicción** son:

* Tabla Resumen:

***Mejor Combinación ira en Negrita y Cursiva***

| Capas               | Learning Rate | Función de Activación | Accuracy       |
| ------------------- | ------------- | ----------------------- | -------------- |
| [50, ]              | 0.001         | logistic                | 0.3378         |
| [50, ]              | 0.001         | tanh                    | 0.5742         |
| [50, ]              | 0.001         | relu                    | 0.5796         |
| [50, ]              | 0.01          | logistic                | 0.6182         |
| [50, ]              | 0.01          | tanh                    | 0.7078         |
| [50, ]              | 0.01          | relu                    | 0.7253         |
| [100, ]             | 0.001         | logistic                | 0.6384         |
| [100, ]             | 0.001         | tanh                    | 0.5846         |
| [100, ]             | 0.001         | relu                    | 0.5944         |
| [100, ]             | 0.01          | logistic                | 0.3912         |
| [100, ]             | 0.01          | tanh                    | 0.5796         |
| [100, ]             | 0.01          | relu                    | 0.7491         |
| [50,  50]          | 0.001         | logistic                | 0.1733         |
| [50,  50]          | 0.001         | tanh                    | 0.8707         |
| [50,  50]          | 0.001         | relu                    | 0.6050         |
| [50,  50]          | 0.01          | logistic                | 0.5247         |
| [50,  50]          | 0.01          | tanh                    | 0.7358         |
| [50,  50]          | 0.01          | relu                    | 0.7708         |
| [100,  50]         | 0.001         | logistic                | 0.1592         |
| [100,  50]         | 0.001         | tanh                    | 0.5767         |
| [100,  50]         | 0.001         | relu                    | 0.6199         |
| [100,  50]         | 0.01          | logistic                | 0.5464         |
| [100,  50]         | 0.01          | tanh                    | 0.7644         |
| ***[100,  50]*** | ***0.01***  | ***relu***            | ***0.7863*** |

* Matriz de Confusión:

![ImagenMatriz](https://github.com/RMTorrabadella04/MIA_Perceptron-Multicapa/blob/master/matriz_confusion_emnist.png)


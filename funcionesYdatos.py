# Con Sklean pillare el dataset EMNIST, Normalizare, Split, Confusion Matrix, Classification Report
# No pillare nada para hacer el modelo, el modelo lo hare a mano con numpy solo.

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler as minmax
from sklearn.model_selection import train_test_split

import numpy as np

# Recogemos los datos y los normalizamos, dividimos en train y test, y devolvemos los datos listos para usar en el modelo.

def datos():

    datos = datasets.fetch_openml('emnist_balanced', version=1, as_frame=False)
    X = datos.data
    y = datos.target.astype(int)

    y_one_hot = np.eye(47)[y]
    
    scaler = minmax()
    X_normalizado = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y_one_hot, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


# Funciones de activación y sus derivadas para el backpropagation

# ReLU y su derivada

def relu(X):
    return np.maximum(0, X)

def relu_derivada(X):
    return (X > 0).astype(float)

# Tanh y su derivada

def tanh(X):
    return np.tanh(X)

def tanh_derivada(X):
    return 1 - np.tanh(X)**2

# Logistic y su derivada

def logistic(X):
    return 1 / (1 + np.exp(-X))

def logistic_derivada(X):
    s = logistic(X)
    return s * (1 - s)

# Softmax para la capa de salida

def softmax(X):
    exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
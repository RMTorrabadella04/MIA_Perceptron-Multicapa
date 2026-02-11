# Con Sklean pillare el dataset EMNIST, Normalizare, Split, Confusion Matrix, Classification Report
# No pillare nada para hacer el modelo, el modelo lo hare a mano con numpy solo.

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler as minmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import numpy as np
import matplotlib.pyplot as plt

# Cargamos el dataset EMNIST

datos = datasets.fetch_openml('emnist_balanced', version=1, as_frame=False)
X = datos.data
y = datos.target

# Normalizamos

scaler = minmax()
X_normalizado = scaler.fit_transform(X)

# Dividimos en train y test

X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.2, random_state=42)



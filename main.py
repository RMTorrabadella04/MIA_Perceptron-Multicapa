# Con Sklean pillare el dataset EMNIST, Normalizare, Split, Confusion Matrix, Classification Report
# No pillare nada para hacer el modelo, el modelo lo hare a mano con numpy solo.

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# La recogida de datos, normalización, división en train y test, y funciones de activación las tengo en otro archivo, 
# lo importo para usarlas aquí.

import funcionesYdatos as fyd

import numpy as np
import matplotlib.pyplot as plt

class PerceptronMulticapa:
    def __init__(self, n_entradas, n_ocultas, n_salidas):
        self.capas = [n_entradas, n_ocultas, n_salidas]
        self.pesos = []
        self.biases = []
    
        for i in range(len(self.capas) - 1):
            peso = np.random.randn(self.capas[i], self.capas[i+1]) * 0.01
            bias = np.zeros((1, self.capas[i+1]))
            
            self.pesos.append(peso)
            self.biases.append(bias)

    # El entrenamiento del modelo

    def training(self, epochs, learning_rate, func_act, func_act_derivada):
        # Recogemos los datos
        
        X_train, X_test, y_train, y_test = fyd.datos()
        

    # La ida (forward)

    def forward(self, X, func_act):
        activaciones = [X]
        zs = []
        
        for i in range(len(self.pesos)):
            z = np.dot(activaciones[-1], self.pesos[i]) + self.biases[i]
            zs.append(z)
            
            if i < len(self.pesos) - 1:
                a = func_act(z)
            else:
                a = fyd.softmax(z)
            
            activaciones.append(a)
        
        return activaciones, zs

    # La vuelta (backpropagation)

    def backpropagation(self, activaciones, zs, y_true, func_act_derivada, lr):
        delta = activaciones[-1] - y_true 
        
        for i in reversed(range(len(self.pesos))):
            dW = np.dot(activaciones[i].T, delta) / activaciones[i].shape[0]
            db = np.mean(delta, axis=0, keepdims=True)
            
            if i > 0:
                delta_proximo = np.dot(delta, self.pesos[i].T) * func_act_derivada(zs[i-1])
            
            self.pesos[i] -= lr * dW
            self.biases[i] -= lr * db
            
            if i > 0:
                delta = delta_proximo

if __name__ == "__main__":
    # Parámetros del entrenamiento
    
    epochs = 10
    learning_rate = 0.01
    func_act = fyd.relu
    func_act_derivada = fyd.relu_derivada
    
    modelo = PerceptronMulticapa(784, 100, 47)
    modelo.training(epochs, learning_rate, func_act, func_act_derivada)
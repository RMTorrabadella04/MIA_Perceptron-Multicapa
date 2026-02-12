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
        
        print("Entrenando el modelo...")
        
        for epoch in range(epochs):
            
            activaciones, zs = self.forward(X_train, func_act)
            
            self.backpropagation(activaciones, zs, y_train, func_act_derivada, learning_rate)
            
            loss = np.mean(np.square(activaciones[-1] - y_train))
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}\n")
            
        return X_test, y_test
    
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

    # Función de predicción

    def predict(self, X, func_act):
        
        activaciones, _ = self.forward(X, func_act)
        
        y_prediccion = activaciones[-1]

        return np.argmax(y_prediccion, axis=1)

if __name__ == "__main__":
    # Parámetros del entrenamiento
    
    epochs = 10
    learning_rate = 0.01
    
    modelo = PerceptronMulticapa(784, 100, 47)
    
    for funcion in [fyd.logistic, fyd.tanh, fyd.relu]:
        # Para que use todas las funciones de activación, sin tener que cambiar el código cada vez.
        
        if funcion == fyd.logistic:
            fun_derivada = fyd.logistic_derivada
        elif funcion == fyd.tanh:
            fun_derivada = fyd.tanh_derivada
        elif funcion == fyd.relu:
            fun_derivada = fyd.relu_derivada

        X_test, y_test = modelo.training(epochs, learning_rate, funcion, fun_derivada)
        
        y_prediccion = modelo.predict(X_test, funcion)
        
        # Matriz de Confusion
        
        y_test_normal = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_test_normal, y_prediccion)
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        disp.plot(cmap='Blues', ax=ax, colorbar=True)
        
        plt.title(f'Matriz de Confusión - {funcion.__name__} - EMNIST Balanced')
        
        plt.show()
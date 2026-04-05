import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

class Perceptron(BaseEstimator, ClassifierMixin):

    def __init__(self, eta=0.01, n_iter=500, bias=0.5):
        # Inicializamos los atributos
        self.eta = eta
        self.n_iter = n_iter
        self.w=[]
        self.bias = bias
    
    def fit(self, X, y):
        # En caso de que se reciba un dataframe, lo pasamos a numpy para poder trabajar con arrays, que es algo más cómodo
        if type(X).__module__ == 'pandas.core.frame' and type(X).__name__ == 'DataFrame':
            X = X.to_numpy()
        if type(y).__module__ == 'pandas.core.series' and type(y).__name__ == 'Series':
            y = y.to_numpy()
        self.w = np.zeros(len(X[0])) # Rellenamos el array de los pesos a 0
        for _ in range(0,self.n_iter): # Se hace las iteracciones hasta 'n_iter'
            # Inicializamos el contador del valor Y a 0 por cada iteración
            # ya que esta nos va a servir para ir obteniendo cada valor de Y por filas
            counterY = 0
            for row in X: # Recorremos cada fila
                # El array de pesos tiene tantos valores como variables X haya,
                # por lo que una opción es poner un contador e ir incrementandolo
                # cada vez que nos desplacemos por la fila.
                # Siempre que se cambie de fila se tiene que establecer de nuevo a 0
                counterWeight = 0
                predict = self.predict_fit(row)
                for valor in row: # Vamos desplazándonos por la fila
                    computer_error = self.eta * (y[counterY] - predict) * valor # Calculamos el error de computación
                    self.w[counterWeight] = self.w[counterWeight] + computer_error # Actualizamos el valor del peso de la variable correspondiente
                    counterWeight+=1 # Aumentamos el contador del peso, ya que ahora nos desplazamos hacia la derecha de la fila
                self.bias += self.eta * (y[counterY] - predict)
                counterY+=1 # En este caso, aumentamos el valor del contador de Y, ya que bajamos una fila, por lo que el valor de Y cambia
    
    def net_input(self, X):
        return np.dot(np.transpose(self.w), X) + self.bias
    
    def predict(self, X, umbral=0.7):
        if type(X).__module__ == 'pandas.core.frame' and type(X).__name__ == 'DataFrame':
            X = X.to_numpy()
        if umbral<0: # Si se establece un umbral menor que 0.5, lo establecemos a 0.5, ya que no tiene sentido un umbral menor de 0
            umbral=0.5
        results = []
        for row in X: # Es posible que se quiera predecir un conjunto de datos, por lo que lo recorremos con un bucle
            p_scalar = self.net_input(row) # Calculamos el producto escalar de los datos a predecir
            sigmoide = expit(p_scalar)
            results.append(1 if sigmoide>umbral else 0) # Agregamos el resultado al array
        return results

    # En este caso no hay un bucle recorriendo el producto escalar, ya que esta función solo se utiliza en el fit
    # y en ese caso no se pasa un conjunto de datos, sino que va fila por fila
    def predict_fit(self, X):
        p_scalar = self.net_input(X)
        sigmoide = expit(p_scalar)
        return sigmoide

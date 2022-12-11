"""
Regresion Lineal por el metodo de Stochastic Gradient Descent
@author:  OTTO F. WAGNER
"""
"""
Comentarios: si divido eta entre el número de observaciones el método tarda
mucho en converger, por tanto prefiero dejarlo sin dividir.

"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score


class SGDLinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self, max_iter=1000, eta=0.01):
        self.max_iter = max_iter
        self.eta = eta
        
        

    def fit(self, X, y):
        X=np.c_[np.ones((X.shape[0],1)),X]

#        n=X.shape[0]
#        if self.max_iter>n:
#            self.max_iter=n #por si nos pasamos
        
        self.beta_ = np.random.uniform(0,1,X.shape[1]) # pesos iniciales

        for inx in range(self.max_iter):
            indice=np.random.randint(0,X.shape[0])
            
            grad = 2 * X[indice,].T * (X[indice,] @ self.beta_ - y[indice])
         
            self.beta_ = self.beta_ - (self.eta) * grad
        

    def predict(self, X):
        
        X=np.c_[np.ones((X.shape[0],1)),X]
        preds=X@self.beta_
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)


from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


X, y = load_boston(return_X_y=True)
sgd_lr = Pipeline([('stds', StandardScaler()), ('sgd_lr', SGDLinearRegression())])
sgd_lr.fit(X, y)
print("R2 SGDLR: " + str(sgd_lr.score(X, y)))
sgd = Pipeline([('stds', StandardScaler()), ('sgd', SGDRegressor())])
sgd.fit(X, y)
print("R2 SGDR: " + str(sgd.score(X, y)))



##########PRUEBAS##############


prb=X2[320:325]

eta = 0.1 
max_iter =10



X2=np.c_[np.ones((X.shape[0],1)),X]

n=X2.shape[0]

#if max_iter>n:
#    max_iter=n
#X2[2,]

beta_ = np.random.uniform(0,1,X2.shape[1]) # pesos iniciales

for inx in range(max_iter):
 
 indice=np.random.randint(0,X2.shape[0])
 grad = 2/n * X2[indice,].T * (X2[indice,] @ beta_ - y[indice])
 
 beta_ = beta_ - (eta/n) * grad
 print(grad)
 
 
X2@beta_


np.random.randint(0,X2.shape[0])

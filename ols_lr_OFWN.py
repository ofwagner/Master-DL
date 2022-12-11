"""
Regresion Lineal por el metodo de Ordinary Least Squares
@author: OTTO F. WAGNER
"""


import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score


class OLSLinearRegression(BaseEstimator, RegressorMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        X2=np.c_[np.ones((X.shape[0],1)),X]
        self.beta_=np.linalg.inv((X2.T@X2))@X2.T@y
        return self
        
    def predict(self, X):
        X2=np.c_[np.ones((X.shape[0],1)),X]
        preds=X2@self.beta_
        return preds

    def score(self, X, y):
        
        preds = self.predict(X)
        return r2_score(y, preds)


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


X, y = load_boston(return_X_y=True)
ols_lr = Pipeline([('stds', StandardScaler()), ('ols_lr', OLSLinearRegression())])
ols_lr.fit(X, y)
print("R2 OLS: " + str(ols_lr.score(X, y)))
lr = Pipeline([('stds', StandardScaler()), ('lr', LinearRegression())])
lr.fit(X, y)
print("R2 LR: " + str(lr.score(X, y)))


###################################
##############PRUEBAS##############
X2=np.c_[np.ones((X.shape[0],1)),X]

(X2.T@X2)^(-1)

beta=np.linalg.inv((X2.T@X2))@X2.T@y

X2@beta_

 
 
 
 
 X2.shape
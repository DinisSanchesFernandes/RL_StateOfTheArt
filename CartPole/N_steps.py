import q_learning
import numpy as np


class SGDRegressor:

    def __init__(self,**kwargs):
        self.w = None
        self.lr = 10e-3
    
    def partial_fit(self,X,Y):

        if self.w is None:

            D = X.shape[1]

            self.w = np.random.randn(D) / np.sqrt(D)

        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self,X):

        return X.dot(self.w)
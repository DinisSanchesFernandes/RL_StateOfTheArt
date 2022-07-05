import numpy as np
import tensorflow as tf
import keras
import q_learning

# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer

class SGDRegressor:
  def __init__(self,D):

    Wtensor = tf.random.uniform(shape = [D,1], dtype = tf.float32)
    Xtensor = tf.random.uniform(shape = [1,D], dtype = tf.float32)
    GTensor = tf.random.uniform(shape = [1], dtype = tf.float32)

    self.X = tf.Variable(Xtensor)
    self.W = tf.Variable(Wtensor)
    self.G = tf.Variable(GTensor)

    self.sgd = tf.keras.optimizers.SGD(learning_rate=0.1)

    self.cost_fn = lambda : (self.G - tf.matmul(self.X,self.W))**2



  def partial_fit(self,X,G):
    
    self.sgd.minimize(self.cost_fn,var_list=[self.W])


  def predict(self,X):
    return tf.reshape(tf.matmul(self.X,self.W),[-1])
  
  

if __name__ == '__main__':

  q_learning.SGDRegressor = SGDRegressor
  q_learning.main()
  #SGD.partial_fit()
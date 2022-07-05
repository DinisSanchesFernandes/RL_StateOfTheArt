import gym 
from gym import wrappers
import numpy as np
import os
import sys
import matplotlib

from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor


class SGDRegressor:

    def __init__(self,D):

        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 10e-2

    def partial_fit(self,X,Y):

        self.w += self.lr * (Y - X.dot(self.w)).dot(X)

    def predict(self,X):

        return X.dot(self.w) 

class FeatureTransformer:

    def __init__(self,env,n_components=1000):

        observation_examples = np.random.random((20000,4)) * 2 -2

        #Aqui está se a definir parametros o "Scalar"
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        #A FeatureUnion servem para concatnar os dados
        #Concatnar os dados de forma a testa varios Kernels de um so vez
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=n_components))
            ]) 
        
        feature_examples = featurizer.fit_transform(scaler.transform(observation_examples))

        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):

        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)


# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer
    for i in range(env.action_space.n):
      model = SGDRegressor(feature_transformer.dimensions)
      self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    result = np.stack([m.predict(X) for m in self.models]).T
    return result

  def update(self, s, a, G):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    self.models[a].partial_fit(X, [G])

  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))


def play_one(env, model, eps, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    if done:
      reward = -200

    # update the model
    next = model.predict(observation)
    # print(next.shape)
    assert(next.shape == (1, env.action_space.n))
    G = reward + gamma*np.max(next)
    model.update(prev_observation, action, G)

    if reward == 1: # if we changed the reward to -200
      totalreward += reward
    iters += 1

  return totalreward

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()

def main():
  env = gym.make('CartPole-v0')
  ft = FeatureTransformer(env)
  model = Model(env, ft)
  gamma = 0.99

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 500
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(env, model, eps, gamma)
    totalrewards[n] = totalreward
    if n % 100 == 0:
      print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())

  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)


if __name__ == '__main__':
  main()
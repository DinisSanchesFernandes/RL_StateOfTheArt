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

class FeatureTransformer:

    def __init__(self,env,n_components=500):

        #Take Random State Samples From Environment
        #These Values Will Be USed To Create the feature transformation
        #The Fit Transformation uses values in the form np.array()
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])     

        #Aqui está se a definir parametros o "Scalar"
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        #A FeatureUnion servem para concatnar os dados
        #Concatnar os dados de forma a testa varios Kernels de um so vez
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ]) 

        featurizer.fit(scaler.transform(observation_examples))

        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):

        scaled = self.scaler.transform(observations)

        return self.featurizer.transform(scaled)

class Model:

    def __init__(self, env, FeatureTransformer, LearningRate):

        self.env = env
        self.models = []
        self.FeatureTransformer = FeatureTransformer

        #Aqui Está se a criar um Modelo para cada uma das ações
        for i in range (env.action_space.n):
            
            # Inicializa o objeto Modelo
            # SGDRegrssor implmenta gradient descent 
            model = SGDRegressor(learning_rate=LearningRate)

            model.partial_fit(FeatureTransformer.transform([env.reset()]) , [0])

            self.models.append(model)
    
    def predict(self,state):

        # State tem de ser uma lista [state] isto pq a API o obriga Scikitlearn
        X = self.FeatureTransformer.transform([state])
        assert(len(X.shape) == 2)
        return np.array([m.predict(X) for m in self.models])

    def update(self,state,action,G_return):

        X = self.FeatureTransformer.transform([state])
        assert(len(X.shape) == 2)
        self.models[action].partial_fit(X,[G_return])

    def sample_action(self,s,eps):

        if np.random.random() < eps:

            return self.env.action_space.sample()
        
        else:

            return np.argmax(self.predict(s))

def play_one(model,env,eps,gamma):

    observation = env.reset()
    done = False
    totalReward = 0
    iters = 0

    while not done and iters < 10000:

        action = model.sample_action(observation,eps)

        prev_observation = observation

        observation, reward, done, _ = env.step(action)

        G = reward + gamma * np.max(model.predict(observation)[0])
        model.update(prev_observation,action,G)

        totalReward += reward
        iters +=1
    
    return totalReward


def plot_cost_to_go(env, estimator, num_tiles=20):
  x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
  y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
  X, Y = np.meshgrid(x, y)
  # both X and Y will be of shape (num_tiles, num_tiles)
  Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
  # Z will also be of shape (num_tiles, num_tiles)

  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, Z,
    rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
  ax.set_xlabel('Position')
  ax.set_ylabel('Velocity')
  ax.set_zlabel('Cost-To-Go == -V(s)')
  ax.set_title("Cost-To-Go Function")
  fig.colorbar(surf)
  plt.show()


def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()


def main(show_plots=True):
  env = gym.make('MountainCar-v0')
  ft = FeatureTransformer(env)
  model = Model(env, ft, "constant")
  gamma = 0.99

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 300
  totalrewards = np.empty(N)
  for n in range(N):
    # eps = 1.0/(0.1*n+1)
    eps = 0.1*(0.97**n)
    if n == 199:
      print("eps:", eps)
    # eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(model, env, eps, gamma)
    totalrewards[n] = totalreward
    if (n + 1) % 100 == 0:
      print("episode:", n, "total reward:", totalreward)
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", -totalrewards.sum())

  if show_plots:
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # plot the optimal state-value function
    plot_cost_to_go(env, model)

if __name__ == '__main__':
  # for i in range(10):
  #   main(show_plots=False)
  main()
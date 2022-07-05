
import gym
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

env = gym.make('MountainCar-v0')

class SGDRegressor:
  def __init__(self, **kwargs):
    self.w = None
    self.lr = 1e-2

  def partial_fit(self, X, Y):
    if self.w is None:
      D = X.shape[1]
      self.w = np.random.randn(D) / np.sqrt(D)
    self.w += self.lr*(Y - X.dot(self.w)).dot(X)

  def predict(self, X):
    return X.dot(self.w)


class FeatureTransformer:
  def __init__(self, env, n_components=500):
    # Retira 10000 Estados aleatorios 
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))

    #print(example_features)

    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer

class Model:
  def __init__(self, env, feature_transformer, learning_rate):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer
    for i in range(env.action_space.n):
      model = SGDRegressor(learning_rate=learning_rate)
      model.partial_fit(feature_transformer.transform( [env.reset()] ), [0])
      self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform([s])
    result = np.stack([m.predict(X) for m in self.models]).T
    assert(len(result.shape) == 2)
    return result

  def update(self, s, a, G):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2)
    self.models[a].partial_fit(X, [G])

  def sample_action(self, s, eps):
    # eps = 0
    # Technically, we don't need to do epsilon-greedy
    # because SGDRegressor predicts 0 for all states
    # until they are updated. This works as the
    # "Optimistic Initial Values" method, since all
    # the rewards for Mountain Car are -1.
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))

def play_one(model,eps,gamma,n=5):

  observation = env.reset()
  done = False
  totalReward = 0
  rewards = []
  states = []
  actions = []
  iters = 0

  # Este Array Guarda os Lambdas
  # [ Gamma^0, Gamma^1, (...), Gamma^T-1]
  multiplier = np.array([gamma]*n)**np.arange(n)
  
  while not done and iters < 10000:

    # Epsilon Greedy
    action = model.sample_action(observation,eps)

    # Store Action and State
    states.append(observation)
    actions.append(action)

    # Execute Action
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    # Store Reward
    rewards.append(reward)

    # Update Model
    # To Update The Model We need at least N rewards
    if len(rewards) >= n:
      
      # Return up to prediction
      # G = R(t+1)*Gamma^0 + R(t+2)*Gamma^1 + ... + R(t+T)*Gamma^T-1
      # rewards[-n:] => Todos os valores 
      return_up_to_prediction = multiplier.dot(rewards[-n:])

      
    


#if __name__ == '__main__':

    #env = gym.make('MountainCar-v0')
    #ft = FeatureTransformer(env)




    

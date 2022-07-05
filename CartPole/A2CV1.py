import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp 
import gym
import numpy as np
from typing import Any, List, Sequence, Tuple


gamma = 0.99
max_steps_per_episode = 200

eps = np.finfo(np.float32).eps.item()
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
env = gym.make("CartPole-v0")




class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self, 
      num_actions: int, 
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)


def UseModel(model,state):

  # The input of the Model has to be a 2 dim tensor
  state = tf.expand_dims(state, 0)

  #print(state)

  Probs, CriticValue = model(state)

  ActorAction = tf.random.categorical(Probs, 1)[0, 0]

  ProbsNormalized = tf.nn.softmax(Probs)

  return ProbsNormalized, ActorAction.numpy(), CriticValue


"""
def env_step(action):

  s, r, done,_ = env.step(action)
  s = tf.convert_to_tensor(s)
  r = tf.convert_to_tensor(r)
  done = tf.convert_to_tensor(done)

  return s, r, done
"""



num_actions = env.action_space.n  # 2
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32), 
          np.array(reward, np.int32), 
          np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action], 
                           [tf.float32, tf.int32, tf.int32])



"""
def run_episode(env,model,max_steps):

  ActorProbs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  CriticValues = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  Rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  
  # Get state 
  initial_state = env.reset()
  initial_state_shape = initial_state.shape
  state = initial_state  
  t = 0
  done = False

  while t < max_steps and not done: 

    Probs, action, Value = UseModel(model,state)

    state, r, done = env_step(action,env)

    # Squeeze removes the extra dimensions of the tensor
    CriticValues = CriticValues.write(t, tf.squeeze(Value))
    # Store the probabilitie of choose the action executed 
    ActorProbs = ActorProbs.write(t, Probs[0, action])
    Rewards = Rewards.write(t, r)

    state.set_shape(initial_state_shape)

    t += 1

  CriticValues = CriticValues.stack()
  ActorProbs = ActorProbs.stack()
  Rewards = Rewards.stack()

  return CriticValues, ActorProbs, Rewards
"""

def run_episode(
    initial_state: tf.Tensor,  
    model: tf.keras.Model, 
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards

"""
def calculate_return(Rewards,gamma,standardize):

  # Get Reward Array Size
  n = tf.shape(Rewards)[0]

  Returns = tf.TensorArray(dtype=tf.float32,size = n)
  G = tf.constant(0.0)
  G_shape = G.shape

  Rewards = tf.reverse(Rewards,[-1])


  for i in tf.range(n):

    r = Rewards[i]
    G = r + gamma * G
    G.set_shape(G_shape)
    Returns = Returns.write(i,G)
  
  Returns = tf.reverse(Returns.stack(),[-1])

  if standardize:
    Returns = ((Returns - tf.math.reduce_mean(Returns)) / (tf.math.reduce_std(Returns) + eps))

  return Returns
"""


def get_expected_return(
    rewards: tf.Tensor, 
    gamma: float, 
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) / 
               (tf.math.reduce_std(returns) + eps))

  return returns

"""

def calculate_loss(Returns,Values,ActionProbs):

  Advantage = Returns - Values

  ActionProbsLog = tf.math.log(ActionProbs)
  ActorLoss = -tf.math.reduce_sum(ActionProbsLog * Advantage)

  CriticLoss = huber_loss(Values,ActorLoss)

  return ActorLoss + CriticLoss

"""

def compute_loss(
    action_probs: tf.Tensor,  
    values: tf.Tensor,  
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss

"""
def train_step(env, model,MaxSteps,opt):

  initial_state = env.reset()

  with tf.GradientTape() as tape:

    CriticValues, ActorProbs, Rewards = run_episode(initial_state,model,MaxSteps)

    Returns = calculate_return(Rewards,gamma,True)

    ActorProbs, CriticValues, Returns = [tf.expand_dims(x, 1) for x in [ActorProbs, CriticValues, Returns]]

    Loss = calculate_loss(Returns,CriticValues,ActorProbs)

  grads = tape.gradient(Loss, model.trainable_variables)

  opt.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(Rewards)

  return episode_reward
"""


@tf.function
def train_step(
    initial_state: tf.Tensor, 
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    gamma: float, 
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode) 

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward





min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 1000

# Cartpole-v0 is considered solved if average reward is >= 195 over 100 
# consecutive trials
reward_threshold = 195
running_reward = 0

# Discount factor for future rewards
gamma = 0.99

num_actions = env.action_space.n  # 2
num_hidden_units = 128



# Keep last episodes reward
for i in range(10000):
  initial_state = tf.constant(env.reset(), dtype=tf.float32)
  episode_reward = int(train_step(initial_state, model, optimizer, gamma, max_steps_per_episode))  
  print(episode_reward)

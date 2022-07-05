import numpy as np
import gym
import tensorflow as tf
import random
import tensorflow.keras.layers as kl
import math

from collections import deque, namedtuple


# This class inits a NN 




class Model(tf.keras.Model):
    """
    Subclassing a multi-layered NN using Keras from Tensorflow
    """
    def __init__(self, num_states, hidden_units, num_actions):
        super(Model, self).__init__() # Used to run the init method of the parent class
        self.input_layer = kl.InputLayer(input_shape = (num_states,))
        self.hidden_layers = []

        for hidden_unit in hidden_units:
            self.hidden_layers.append(kl.Dense(hidden_unit, activation = 'tanh')) # Left kernel initializer
        
        self.output_layer = kl.Dense(num_actions, activation = 'linear')

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output

class ReplayMemory():
    """
    Used to store the experience genrated by the agent over time
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    # This method fills the Replay buffer
    # If the Buffer is full it starts overwriting from the beggining

    def push(self, experience):
        if len(self.memory)<self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    # Return random sample from the replay buffer

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # If Buffer not full this return False

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
    """
    Decaying Epsilon-greedy strategy
    """
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1*current_step*self.decay)

class DQN_Agent():
    """
    Used to take actions by using the Model and given strategy.
    """
    def __init__(self, strategy, num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            return random.randrange(self.num_actions), rate, True
        else:
            return np.argmax(policy_net(np.atleast_2d(np.atleast_2d(state).astype('float32')))), rate, False


def Make_prediction(NN,state):

    return np.argmax(NN(np.array([state])))

def copy_weights(Copy_from, Copy_to):
    """
    Function to copy weights of a model to other
    """
    variables2 = Copy_from.trainable_variables
    variables1 = Copy_to.trainable_variables
    for v1, v2 in zip(variables1, variables2):
        v1.assign(v2.numpy())

env = gym.make('CartPole-v0')

# NN
InputLayerSize = len(env.observation_space.sample())
HiddenLayers = [200,200]
OutputLayersize = env.action_space.n

# Memory Parameters
memory_size = 100000
batch_size = 64

# Hyperparameters
epochs = 10000
timesteps = 200
lr = 0.01
eps_start = 1
eps_end = 0.000
eps_decay = 0.001
target_update = 25
gamma = 0.99

# Init NN 
policy_net = Model(InputLayerSize,HiddenLayers,OutputLayersize)
target_net = Model(InputLayerSize,HiddenLayers,OutputLayersize) 
optimizer = tf.optimizers.Adam(lr)

# Init Memory
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = DQN_Agent(strategy, env.action_space.n)
memory = ReplayMemory(memory_size)

# Init Tuple Var
Experience = namedtuple('Experience', ['states','actions', 'rewards', 'next_states','dones'])

copy_weights(policy_net,target_net)

# Loop Episodes
for epoch in range(epochs):

    # Clean Variables
    done = False
    timestep = 0
    s = env.reset()    
    total_reward = 0

    # Loop Iterations
    while timestep < timesteps and done == False:

        timestep += 1

        a, rate, flag = agent.select_action(s, policy_net)

        #print(a)

        s1, r, done, _ = env.step(a)

        total_reward += r

        memory.push(Experience(s,a,s1,r,done))

        s = s1

        if memory.can_provide_sample(batch_size):

            # Get a batch of past experiences
            experience = memory.sample(batch_size)
            batch =Experience(*zip(*experience))

            # Set all states rewards next state and rewards in diferent list variables
            states, actions, rewards, next_states, dones = np.asarray(batch[0]),np.asarray(batch[1]),np.asarray(batch[3]),np.asarray(batch[2]),np.asarray(batch[4])

            # Temporal Diference
            # For each next state get the Max value from the target NN
            q_s_a_prime = np.max(target_net(np.atleast_2d(next_states).astype('float32')), axis = 1)
            
            # For every non terminal state calculate Return 
            q_s_a_target = np.where(dones, rewards, rewards+gamma*q_s_a_prime)

            # Convert the calculated value to a tensor
            q_s_a_target = tf.convert_to_tensor(q_s_a_target, dtype = 'float32')

            with tf.GradientTape() as tape:

                q_s_a = tf.math.reduce_sum(policy_net(np.atleast_2d(states).astype('float32')) * tf.one_hot(actions,env.action_space.n), axis = 1)
                loss = tf.math.reduce_mean(tf.square(q_s_a_target - q_s_a))

            variables = policy_net.trainable_variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            if timestep%target_update == 0:
                copy_weights(policy_net, target_net)

        s = s1

    print("Epoch: ", epoch," Total Reward: ",total_reward)




#main()
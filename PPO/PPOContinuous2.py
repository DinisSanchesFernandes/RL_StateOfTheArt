from cmath import pi
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras as K
import scipy.signal
import gym


env = gym.make("Pendulum-v1")
state_space_dimensions = env.observation_space.shape[0]
num_actions = 1
s = env.reset()

print("------------------------------------------Pendulum Environment------------------------------------------")

print("Dim State Space: ",state_space_dimensions)
print("Dim Action Space: ",num_actions)


# x is the input layers
# sizes a list of all the hidden layers [64, 64] => 2 layers of 64 neurons
def create_neural_network(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


# Init Actor
state = K.layers.Input(shape=(state_space_dimensions,), name='state_input')
dense = K.layers.Dense(32, activation='relu', name='dense1')(state)
dense = K.layers.Dense(32, activation='relu', name='dense2')(dense)
mu = K.layers.Dense(1, activation='tanh',name="actor_output_mu")(dense)
sigma = K.layers.Dense(1, activation='softplus', name="actor_output_sigma")(dense)
mu_and_sigma = K.layers.concatenate([mu, sigma])
actor = K.Model(inputs=state, outputs=mu_and_sigma)
actor_lr = 3e-4
actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)


#actor_nn_hiddenlayers = [64,64]
#common_input = keras.Input(shape=(state_space_dimensions,), dtype=tf.float32)
#actor_output = create_neural_network(common_input,actor_nn_hiddenlayers + [2],tf.tanh,None)
#actor = keras.Model(inputs= common_input, outputs=actor_output)

# define input layer
state = K.layers.Input(shape=(state_space_dimensions,), name='state_input')
# define hidden layers
dense = K.layers.Dense(32, activation='relu', name='dense1')(state)
dense = K.layers.Dense(32, activation='relu', name='dense2')(dense)
# connect the layers to a 1-dim output: scalar value of the state (= Q value or V(s))
V = tf.squeeze(K.layers.Dense(1, name="actor_output_layer")(dense), axis=-1)
# make keras.Model
critic = K.Model(inputs=state, outputs=V)
critic_lr = 1e-3
critic_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)

#Init Critic
#critic_nn_hiddenlayers = [64,64]
#critic_output = tf.squeeze(create_neural_network(common_input, actor_nn_hiddenlayers + [1], tf.tanh, None), axis=-1)
#critic = keras.Model(inputs= common_input, outputs= critic_output)
#critic_lr = 1e-3
#critic_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Buffer:

    def __init__(self, state_space_dimensions, size, gamma = 0.99, lam = 0.95):

        self.reward_buffer = np.zeros(size, dtype = np.float32)
        self.value_buffer = np.zeros(size, dtype = np.float32)
        self.state_buffer = np.zeros((size, state_space_dimensions), dtype = np.float32)
        self.action_buffer = np.zeros(size, dtype = np.float32)
        self.mu_sigma_buffer = np.zeros((size,2), dtype = np.float32)
        
        self.return_buffer = np.zeros(size, dtype = np.float32)
        self.advantage_buffer = np.zeros(size, dtype = np.float32)
        self.gamma = gamma
        self.lam = lam
        self.pointer, self.trajectory_start_index = 0,0

    def get(self):

        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean = np.mean(self.advantage_buffer)
        advantage_deviation = np.std(self.advantage_buffer)

        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_deviation

        return(
            self.state_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.mu_sigma_buffer,
        ) 

    def store(self, r, V_s, s, a, mu_sigma):

        self.reward_buffer[self.pointer] = r
        self.value_buffer[self.pointer] = V_s
        self.state_buffer[self.pointer] = s
        self.action_buffer[self.pointer] = a
        self.mu_sigma_buffer[self.pointer] = mu_sigma

        self.pointer += 1

    def finish_trajectory(self, r_terminal = 0):

        sliced_index = slice(self.trajectory_start_index, self.pointer)

        r = np.append(self.reward_buffer[sliced_index], r_terminal)

        # No followinf state so:
        # V = r + gamma V'
        # V' = 0
        # V = r
        V = np.append(self.value_buffer[sliced_index], r_terminal)

        deltas = r[:-1] + self.gamma * V[1:] - V[:-1]

        self.advantage_buffer[sliced_index] = discounted_cumulative_sums(deltas, self.lam * self.gamma)

        self.return_buffer[sliced_index] = discounted_cumulative_sums(r, self.gamma)[:-1]

        self.trajectory_start_index = self.pointer

def sample_action(s_reshaped):


    mu_sigma = actor(s_reshaped)
    
    mu_sigma = mu_sigma.numpy()[0]

    mu = mu_sigma[0]

    action_mu = mu * 2

    sigma = mu_sigma[1]

    if exploitation:

        action = mu

    else:

        action = np.random.normal(loc = action_mu, scale = sigma, size = 1)

    return action, [mu, sigma]



@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

@tf.function
def train_actor(
    observation_buffer, action_buffer, log_PDF_buffer, advantage_buffer
):

    clip_ratio = 0.2
    print("train")


    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        

        ratio = tf.exp(
            calculate_log_PDF(actor(observation_buffer), action_buffer)
            - log_PDF_buffer
        )
        cliped_ratio = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        actor_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, cliped_ratio)
        )
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    #actions, mus_and_sigmas = actor(observation_buffer)

    kl = tf.reduce_mean(
        log_PDF_buffer
        - calculate_log_PDF(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl

@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

@tf.function
def calculate_log_PDF(mu_and_sigma,action):

    mu = tf.gather(mu_and_sigma, 0, axis = 1)

    sigma = tf.gather(mu_and_sigma, 1, axis = 1)

    action = tf.convert_to_tensor(action_buffer,dtype=tf.float32)

    variance = tf.square(sigma)
    pdf = 1. / tf.sqrt(2. * np.pi * variance) * tf.exp(-tf.square(action - mu) / (2. * variance))
    log_pdf = tf.math.log(pdf + 1e-7)
    #print("LOLOLOLO")
    return log_pdf

# Hyperparameters
exploitation = False
total_epochs = 1000
epoch_steps = 30
train_policy_iterations = 80
train_value_iterations = 80
target_kl = 0.01
total_episode_reward = 0
render_epoch = 10

Memory = Buffer(state_space_dimensions, epoch_steps)

for epoch in range(total_epochs):

    for step in range(epoch_steps):

        s = s.reshape(1,-1)

        

        a, mu_and_deviation = sample_action(s)

        #print(a)

        if epoch % render_epoch == 0:
            env.render()

        s1, r, done, _ = env.step(a)

        total_episode_reward += r

        V_s = critic(s)

        Memory.store(r,V_s,s,a,mu_and_deviation)

        s = s1

        if done or step == epoch_steps - 1:

            #print("End Epoch")

            last_value = 0 if done else critic(s.reshape(1,-1))
            Memory.finish_trajectory(last_value)
            s = env.reset()
            print("Epoch: ", epoch,"Total Reward: ", total_episode_reward)
            total_episode_reward = 0

    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        mu_and_sigma_buffer,
    ) = Memory.get()

    mu_and_sigma_buffer = tf.convert_to_tensor(mu_and_sigma_buffer)

    #print("Obs: ",observation_buffer)
    #print("Act: ",action_buffer)
    #print("Adv: ",advantage_buffer)
    #print("Ret: ",return_buffer)
    #print("MuSig: ",mu_and_sigma_buffer)
    
    #actions, mus_and_sigmas = actor(observation_buffer)
    #calculate_log_PDF(mus_and_sigmas[0], mus_and_sigmas[1], action_buffer)
    log_PDF_buffer = calculate_log_PDF(mu_and_sigma_buffer,action_buffer)

    #print("musigma",mu_and_sigma_buffer)
    #print("Action", action_buffer)

    #print(log_PDF_buffer)

    for _ in range(train_policy_iterations):
        
        kl = train_actor(
            observation_buffer, action_buffer, log_PDF_buffer, advantage_buffer
        )
       # print("Train cycle", kl)

        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, return_buffer)



exploitation = True
max_steps = 200


while 1:

    s = env.reset()
    done = False

    while step < (max_steps) and not done:

        s = s.reshape(1,-1)

        a, mu_and_deviation = sample_action(s)


        a = [a]

        #print(a)

        env.render()

        s1, r, done, _ = env.step(a)

        total_episode_reward += r

        #V_s = critic(s)

        #Memory.store(r,V_s,s,a,mu_and_deviation)

        s = s1







































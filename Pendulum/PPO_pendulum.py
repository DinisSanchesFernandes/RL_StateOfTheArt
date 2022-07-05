import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal
import gym

env = gym.make("Pendulum-v1")
state_space_dimensions = env.observation_space.shape[0]
action_space_dimensions = 1
s = env.reset()

print("Mountain Car Continuous: ")
print("State Space Dim: ", state_space_dimensions)
print("Action Space Dim: ", action_space_dimensions)


def create_neural_network(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


# Init Actor
actor_nn_hiddenlayers = [400,400]
common_input = keras.Input(shape=(state_space_dimensions,), dtype=tf.float32)
actor_output = create_neural_network(common_input,actor_nn_hiddenlayers + [action_space_dimensions],tf.tanh,None)
actor = keras.Model(inputs= common_input, outputs=actor_output)
actor_lr = 3e-4
actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)


#Init Critic
critic_nn_hiddenlayers = [400,400]
critic_output = tf.squeeze(create_neural_network(common_input, actor_nn_hiddenlayers + [1], tf.tanh, None), axis=-1)
critic = keras.Model(inputs= common_input, outputs= critic_output)
critic_lr = 1e-3
critic_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

@tf.function
def train_actor(
    observation_buffer, action_buffer, advantage_buffer
):

    clip_ratio = 0.2


    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = actor(observation_buffer) - action_buffer
        
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

    kl = tf.reduce_mean(action_buffer - actor(observation_buffer))

    kl = tf.reduce_sum(kl)
 
    return kl

@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

class Buffer:

    def __init__(self,state_space_dimensions, size, gamma = 0.99, lam = 0.95):

        self.reward_buffer = np.zeros(size, dtype = np.float32)
        self.value_buffer = np.zeros(size, dtype = np.float32)
        self.state_buffer = np.zeros((size, state_space_dimensions), dtype = np.float32)
        self.action_buffer = np.zeros(size, dtype = np.float32)
        self.return_buffer = np.zeros(size, dtype = np.float32)
        self.advantage_buffer = np.zeros(size, dtype = np.float32)
        self.gamma = gamma
        self.lam = lam
        self.pointer, self.trajectory_start_index = 0,0

    def store(self, r, V_s, s, a):

        self.reward_buffer[self.pointer] = r
        self.value_buffer[self.pointer] = V_s
        self.state_buffer[self.pointer] = s
        self.action_buffer[self.pointer] = a

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
        )     


# Hyperparameters
epochs = 30
epoch_steps = 1000
target_kl = 0.01
train_policy_iterations = 80
train_value_iterations = 80

Total_reward_ep = 0

render_f = 0
done_cnt = 0


buffer_obj = Buffer(state_space_dimensions, epoch_steps)


for epoch in range(epochs):

    for t in range(epoch_steps):

        s_reshaped = s.reshape(1,-1)

        env.render()

        # Get Action
        a = actor(s_reshaped).numpy() * 2

        # Step
        s1, reward_step, done, _ = env.step(a)
        Total_reward_ep += reward_step

        #print(reward_step)

        # Get Critic Value
        V_s = critic(s_reshaped).numpy()

        buffer_obj.store(reward_step, V_s, s_reshaped, a)

        s = s1

        if done or t == epoch_steps - 1: 

            print("Finished Episode")

            last_value = 0 if done else critic(s.reshape(1,-1))
                
            buffer_obj.finish_trajectory(last_value)

            print("TotalReward: ",Total_reward_ep)

            Total_reward_ep = 0

            s = env.reset()
                
    Total_reward_ep = 0
        
    s_buffer, a_buffer, Adv_buffer, G_buffer = buffer_obj.get()

    print("Train Actor: ")

    for _ in range(train_policy_iterations):
        kl = train_actor(
            s_buffer, a_buffer, Adv_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    print("Train Critic: ")

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(s_buffer, G_buffer)

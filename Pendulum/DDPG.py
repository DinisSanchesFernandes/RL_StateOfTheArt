from email.policy import Policy
import tensorflow as tf
from tensorflow.keras import layers
import gym 
import numpy as np

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

  
def get_CriticModel(num_actions,num_states):

    InputActions = layers.Input(shape=(num_actions))
    HiddenLayerActions = layers.Dense(16,activation = "relu")(InputActions)


    InputStates = layers.Input(shape=(num_states))
    HiddenLayerStates = layers.Dense(16,activation = 'relu')(InputStates)
    HiddenLayerStates = layers.Dense(32,activation = 'relu')(HiddenLayerStates)

    ConcatenatedLayer = layers.Concatenate()([HiddenLayerStates,HiddenLayerActions])

    HiddenLayer = layers.Dense(256,activation = 'relu')(ConcatenatedLayer)
    HiddenLayer = layers.Dense(256,activation = 'relu')(HiddenLayer)

    Out = layers.Dense(1)(HiddenLayer)

    return tf.keras.Model([InputStates,InputActions], Out)

def get_ActorModel(num_states):

    # The initializer is used for the first values not be -1 or 1
    # Which would squash our gradients to zero, as we use the tanh activation
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    InputState = layers.Input(shape=(num_states,))
    HiddenLayer = layers.Dense(256,activation = "relu")(InputState)    
    HiddenLayer = layers.Dense(256,activation = "relu")(HiddenLayer)
    Output = layers.Dense(1,activation = "tanh", kernel_initializer=last_init)(HiddenLayer)

    Output = Output * 2.0
    model = tf.keras.Model(InputState,Output)
    return model

class Buffer():

    def __init__ (self, BufferCapacity, BatchSize, NumStates, NumActions):

        self.BufferCapacity = BufferCapacity
        self.BatchSize = BatchSize  
        self.BufferCount = 0
        self.NumStates = NumStates
        self.NumActions = NumActions


        self.StateBuffer = np.zeros((self.BufferCapacity, self.NumStates))
        self.ActionBuffer = np.zeros((self.BufferCapacity, self.NumActions))
        self.RewardBuffer = np.zeros((self.BufferCapacity,1))
        self.NextStateBuffer = np.zeros((self.BufferCapacity,self.NumStates))

    def store_tuple(self,Tuple):

        index = self.BufferCount % self.BufferCapacity

        self.StateBuffer[index] = Tuple[0]
        self.ActionBuffer[index] = Tuple[1] 
        self.RewardBuffer[index] = Tuple[2]
        self.NextStateBuffer[index] = Tuple[3]

        self.BufferCount += 1

    @tf.function
    def train_actorNN_criticNN(self,StateBatch, ActionBatch, RewardBatch,NextStateBatch):

        # Train Critic
        with tf.GradientTape() as tape:
            
            # Predict Actions For Each State
            PredictActions = ActorNN_Target(NextStateBatch, training = True)

            # Calculate Return using Critic Target NN 
            # V(s,a) = NNtc(s,a)
            # G = r + gamma * V(s,a)
            G = RewardBatch + Gamma * CriticNN_Target([NextStateBatch, PredictActions], training = True)

            # Predict return with Critic NN
            # Ĝ = NNc(s,a)
            G_hat = CriticNN([StateBatch,ActionBatch],training = True)

            # Calculate Loss
            # Loss = G -Ĝ
            LossCritic = tf.math.reduce_mean(tf.math.square(G-G_hat))

        CriticGrads = tape.gradient(LossCritic, CriticNN.trainable_variables)
        CriticOpt.apply_gradients(zip(CriticGrads,CriticNN.trainable_variables))

        with tf.GradientTape() as tape:

            Actions = ActorNN(StateBatch, training = True)

            ActionsValues = CriticNN([StateBatch, Actions], training = True)

            # This value has to be maximized so Loss = -value
            ActorLoss = -tf.math.reduce_mean(ActionsValues)

        ActorGrads = tape.gradient(ActorLoss, ActorNN.trainable_variables)
        ActorOpt.apply_gradients(zip(ActorGrads, ActorNN.trainable_variables))


    def learn(self):

        limits = min(self.BufferCount,self.BufferCapacity)
        BatchIndices = np.random.choice(limits, self.BatchSize)

        StateBatch = tf.convert_to_tensor(self.StateBuffer[BatchIndices])
        ActionBatch = tf.convert_to_tensor(self.ActionBuffer[BatchIndices])
        RewardBatch = tf.convert_to_tensor(self.RewardBuffer[BatchIndices])
        RewardBatch = tf.cast(RewardBatch, dtype=tf.float32)
        NextStateBatch = tf.convert_to_tensor(self.NextStateBuffer[BatchIndices])

        self.train_actorNN_criticNN(StateBatch, ActionBatch, RewardBatch, NextStateBatch)

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def policy_action(State, NoiseObject):

    SampledAction = tf.squeeze(ActorNN(State))

    noise = NoiseObject()

    SampledAction = SampledAction.numpy() + noise

    legal_action = np.clip(SampledAction, lower_bound, upper_bound)

    return[np.squeeze(legal_action)]


# Hyperparameters
Gamma = 0.99
critic_lr = 0.002
actor_lr = 0.001
Episode = 100
tau = 0.005

# Init
problem = "Pendulum-v1"
env = gym.make(problem)

# Noise Object
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

# Display Env Data
num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

# Init NN
CriticNN = get_CriticModel(num_actions,num_states)
CriticNN_Target = get_CriticModel(num_actions,num_states)
CriticOpt = tf.keras.optimizers.Adam(critic_lr)

ActorNN = get_ActorModel(num_states)
ActorNN_Target = get_ActorModel(num_states)
ActorOpt = tf.keras.optimizers.Adam(actor_lr)

# Init Buffer
buffer = Buffer(50000, 64,num_states,num_actions)

for Epis in range(Episode):

    s = env.reset()
    TotalReward = 0

    while True:

        tf_s = tf.expand_dims(tf.convert_to_tensor(s),0)

        a = policy_action(tf_s,ou_noise)

        s1, r, done, _ = env.step(a)

        buffer.store_tuple((s, a, r, s1))

        TotalReward += r

        buffer.learn()
        update_target(ActorNN_Target.variables, ActorNN.variables, tau)
        update_target(CriticNN_Target.variables, CriticNN.variables, tau)

        if done:
            break

        s = s1

    print("Episode: ",Epis, "Total Reward: ",TotalReward)
    TotalReward = 0

while 1:

    s = env.reset()
    TotalReward = 0

    while True:

        env.render()

        tf_s = tf.expand_dims(tf.convert_to_tensor(s),0)

        a = policy_action(tf_s,ou_noise)

        s1, r, done, _ = env.step(a)

        if done:
            break

        s = s1
        
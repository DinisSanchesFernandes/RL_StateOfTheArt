# Import Libraries
import tensorflow as tf
import tensorflow_probability as tfp 
import gym 
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
gamma = 1

# Init Variables
env = gym.make('CartPole-v0')
s = env.reset()

# Policy Neural Network
# Predict => π (s,a) with NN

# NeuralNetworkParameters
NumberOfHiddenLayers = 30

# Making Output Same Size As Action Space 
NumberOfOutputLayer = env.action_space.n

# Define model 
class Pmodel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(NumberOfHiddenLayers,activation='relu')
        self.d2 = tf.keras.layers.Dense(NumberOfHiddenLayers,activation='relu')
        self.out = tf.keras.layers.Dense(NumberOfOutputLayer,activation='softmax')

    def call(self, input_data):
        # The input has to be a tensor
        x = tf.convert_to_tensor(input_data)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x

Popt = tf.keras.optimizers.Adam(learning_rate=0.001)

Pmodel = Pmodel()

# Get Action
# Probability of Each Action Under The Current Policy
def PolicyAction(state):

    # Retirada a Probabilidade para cada uma das ações
    # Uses Neural Network
    prob = Pmodel(np.array([state]))
        
    # Criar um distribuição com as probabilidades anteriores
    dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        
    # Da distribuição retirar uma ação em formato Tensor
    action = dist.sample()
        
    # Passar para integer
    return int(action.numpy()[0])

PolicyLossList = []
# Loss Function
def PolicyLossFunction(prob, action, reward):

    # Creates Tenso Flow Variable With Probabilities Taken From NNP    
    dist = tfp.distributions.Categorical(probs = prob, dtype = tf.float32)
    
    # Gets Logarithmic probability
    log_prob = dist.log_prob(action)

    # Calculates loss
    loss = -log_prob * reward

    #print("Policy loss: ",loss.numpy())

    PolicyLossList.append(loss)

    return loss

def UpdatePolicyNeuralNetwork(state,action,reward):

    with tf.GradientTape() as Ptape:

        p = Pmodel(np.array([state]), training=True)
        Ploss = PolicyLossFunction(p,action,reward)

    Pgrads = Ptape.gradient(Ploss,Pmodel.trainable_variables)
    #print("Pgrads: ",Pgrads)
    Popt.apply_gradients(zip(Pgrads,Pmodel.trainable_variables))

# Value Neural Network
# Predict => V(s)

# NeuralNetworkParameters

# Number Of Input Neurons = Dimensions Given By Feature Transformer for any state
NumberOfHiddenLayers = 30

# Number of output Layers
NumberOfOutputLayer = 1

# Define model 
class Vmodel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(NumberOfHiddenLayers,activation='relu')
        #self.d2 = tf.keras.layers.Dense(NumberOfHiddenLayers,activation='relu')
        self.out = tf.keras.layers.Dense(NumberOfOutputLayer,activation='relu')

    def call(self, input_data):
        # The input has to be a tensor
        x = tf.convert_to_tensor(input_data)
        x = self.d1(x)
        #x = self.d2(x)
        x = self.out(x)
        return x

Vopt = tf.keras.optimizers.SGD(learning_rate=0.0001)

Vmodel = Vmodel()

def PredictValue(state):

    #state2D = np.atleast_2d(state)
    
    return Vmodel(np.array([state]))

#print("Predict V: ",PredictValue(s))

ValueLossList = []

def ValueLossFunction(G,G_hat):

    erro = G - G_hat

    loss = tf.reshape(tf.convert_to_tensor(tf.square(erro)), [-1])

    #loss = tf.convert_to_tensor(tf.reduce_sum(tf.square(G - G_hat)))   

    #print("Value Loss: ",loss.numpy())

    #print(G.numpy()," - ",G_hat.numpy()," loss: ",loss.numpy())

    ValueLossList.append(loss)

    return loss

#print("Loss: ",ValueLossFunction(2))


def UpdateValueNeuralNetwork(G,state):

    with tf.GradientTape() as Vtape:

        G_hat = Vmodel(np.array([state]), training=True)
        Vloss = ValueLossFunction(G,G_hat)

    Vgrads = Vtape.gradient(Vloss,Vmodel.trainable_variables)
    #print("Vgrads: ",Vgrads)
    Vopt.apply_gradients(zip(Vgrads,Vmodel.trainable_variables))

# Main
N = 500
Env_rewards = []

for n in range(N):

    done = False
    iters = 0
    env.reset()
    total_reward = 0
    total_envreward = 0

    while not done and iters < 2000:

        # Get Action  Action <= π (s,a)
        a = PolicyAction(s)

        # Step (Action)
        s1, r, done, _ = env.step(a)

        # Env Reward
        total_envreward += r

        #r = r * iters


        # Predict V(s1) <= NNV(s1)
        V_s1 = PredictValue(s1)

        # G = r + gamma * V(s1)
        G = r + gamma * V_s1
        #print("R: ",r,"G: ",G.numpy(),"Vs1",V_s1.numpy(),"Action: ",a)

        # Register Reward Episode
        total_reward += r

        # Partial_fit NNP
        UpdatePolicyNeuralNetwork(s,a,r)

        # Partial_fit NNV
        UpdateValueNeuralNetwork(G,s)

        s = s1

        iters += 1

    Env_rewards.append(total_reward)

    print("Episode: ", iters,"Total Reward: ", total_envreward)

plt.plot(ValueLossList, label = 'ValueLoss')
plt.plot(Env_rewards, label = 'Reward')
plt.plot(PolicyLossList, label = 'PolicyLossList')
plt.legend()
plt.show()


#print("Hello")
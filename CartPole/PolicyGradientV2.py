import tensorflow as tf
import tensorflow_probability as tfp 
import gym 
import numpy as np

#https://github.com/abhisheksuran/Reinforcement_Learning/blob/master/Reinforce_(PG)_ReUploaded.ipynb
#https://towardsdatascience.com/reinforce-policy-gradient-with-tensorflow2-x-be1dea695f24

env = gym.make('CartPole-v0')

# Define model 
class Pmodel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(30,activation='relu')
        self.d2 = tf.keras.layers.Dense(30,activation='relu')
        self.out = tf.keras.layers.Dense(env.action_space.n,activation='softmax')

    def call(self, input_data):
        # The input has to be a tensor
        x = tf.convert_to_tensor(input_data)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x

# Define model 
class Vmodel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(30,activation='relu')
        self.d2 = tf.keras.layers.Dense(10,activation='relu')
        self.out = tf.keras.layers.Dense(1,activation='relu')

    def call(self, input_data):
        # The input has to be a tensor
        x = tf.convert_to_tensor(input_data)
        x = self.d1(x)
        x = self.d2(x)
        x = self.out(x)
        return x

PolicyLossList = []

class agent():

    def __init__(self):
    
        self.Pmodel = Pmodel()
        self.Vmodel = Vmodel()
        self.Popt = tf.keras.optimizers.Adagrad(learning_rate=0.001)
        self.Vopt = tf.keras.optimizers.SGD(learning_rate=0.001)
        self.gamma = 1

    def act(self,state):

        # Retirada a Probabilidade para cada uma das ações
        # Uses Neural Network
        prob = self.Pmodel(np.array([state]))
        
        # Criar um distribuição com as probabilidades anteriores
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        
        # Da distribuição retirar uma ação em formato Tensor
        action = dist.sample()
        
        # Passar para integer
        return int(action.numpy()[0])

    def P_loss(self, prob, action, error):
        
        dist = tfp.distributions.Categorical(probs = prob, dtype = tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * error
        PolicyLossList.append(loss)
        return loss

    def trainPolicy(self, state, action, error):

        with tf.GradientTape() as tape:

            p = self.Pmodel(np.array([state]), training=True)
            loss = self.P_loss(p,action,error)

        grads = tape.gradient(loss,self.Pmodel.trainable_variables)
        self.Popt.apply_gradients(zip(grads,self.Pmodel.trainable_variables))


    def Predict(self,state):

        #state = np.atleast_2d(state)

        return self.Vmodel(np.array([state]))

    def trainValueFunc(self,state,G):

        with tf.GradientTape() as tape:

            #state = np.atleast_2d(state)
            G_hat = self.Vmodel(np.array([state]),training = True)
            loss = tf.square(G - G_hat)

        grads = tape.gradient(loss,self.Vmodel.trainable_variables)
        self.Vopt.apply_gradients(zip(grads,self.Vmodel.trainable_variables))


    def TrainMC(self, states, actions, rewards):

        G = 0

        errlist = []
        Glist = []

        for s,r,a in zip(reversed(states),reversed(rewards),reversed(actions)):

            errlist.append(G - self.Predict(s)) 

            G = r + self.gamma * G
            Glist.append(G)

        self.trainPolicy(states,actions,errlist)

        self.trainValueFunc(states,Glist)




N = 500
max_iters = 200
gamma = 0.6

CartAgent = agent()

for n in range(N):

    iters = 0
    done = False
    totalReward = 0

    s = env.reset()

    states = []
    actions = []
    rewards = []
    r = 0

    while (iters < max_iters) and not done:

        a = CartAgent.act(s)

        states.append(s)
        actions.append(a)
        rewards.append(r)

        s, r, done,__ = env.step(a)

        totalReward += r
    
    states.append(s)
    actions.append(a)
    rewards.append(r)

    CartAgent.TrainMC(states,actions,rewards)

    print("Ep: ",n,"TotalReward: ",totalReward)
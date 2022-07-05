import tensorflow as tf
import tensorflow_probability as tfp 
import gym 
import numpy as np

#https://github.com/abhisheksuran/Reinforcement_Learning/blob/master/Reinforce_(PG)_ReUploaded.ipynb
#https://towardsdatascience.com/reinforce-policy-gradient-with-tensorflow2-x-be1dea695f24

env = gym.make('MountainCar-v0')

# Define model 
class model(tf.keras.Model):
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


class agent():

    def __init__(self):
    
        self.model = model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 1

    def act(self,state):

        # Retirada a Probabilidade para cada uma das ações
        # Uses Neural Network
        prob = self.model(np.array([state]))
        
        # Criar um distribuição com as probabilidades anteriores
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        
        # Da distribuição retirar uma ação em formato Tensor
        action = dist.sample()
        
        # Passar para integer
        return int(action.numpy()[0])

    def a_loss(self, prob, action, reward):
        
        dist = tfp.distributions.Categorical(probs = prob, dtype = tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward
        return loss

    def train(self, states, rewards, actions):

        sum_reward = 0
        rewards.reverse()
        discnt_rewards = []

        for r in rewards:
            
            sum_reward = r + self.gamma * sum_reward

            discnt_rewards.append(sum_reward)

        discnt_rewards.reverse()

        for state,reward,action in zip(states,discnt_rewards,actions):

            with tf.GradientTape() as tape:

                p = self.model(np.array([state]), training=True)
                loss = self.a_loss(p,action,reward)

            grads = tape.gradient(loss,self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads,self.model.trainable_variables))




        #print("Train")        

def reward_function(NextState):

    return np.power(50,NextState-0.2)



Agent = agent()

Render_F = True


#Hyperparameters
steps = 500

# Loop Episodes
for s in range(steps):

    state = env.reset()

    rewards = []
    actions = []
    states = []
    total_reward = 0
    done = False




    #Loop Iterations
    while not done:

        action = Agent.act(state)

        next_state, reward_Gym, done, _ = env.step(action)

        #print("Altitude: ", next_state[0], "Reward: ",reward_function(next_state[0]))

        reward = reward_function(next_state[0])

        if Render_F:

            env.render()

        rewards.append(reward)
        states.append(state)
        actions.append(action)

        state = next_state

        total_reward += reward


    if done:

        Agent.train(states,rewards,actions)

        print("total reward after {} steps is {}".format(s, total_reward))

        #if total_reward >= 200:

        #    Render_F = True
import random as rand
import numpy
import matplotlib.pyplot as plt

BANDITS_PROB = [0.2,0.5,0.75]
EPISODES = 10000
EPSILON = 0.1

class BanditModel():

    def __init__(self,p):

        self.p = p
        self.estimatedReward = 0
        self.N = 0


    def PullFunc(self):

        return rand.random() > self.p

    def UpdateModel(self, X):

        self.N = self.N + 1

        self.estimatedReward = ((self.N - 1)* self.estimatedReward + X) * (1 / self.N)

        
def Experiment():
    
    num_times_explored = 0
    num_times_exploited = 0

    #Create Array to register all the rewards in each episode
    rewards = numpy.zeros(EPISODES)

    #Passar o array de BANDITS_PROB para o array class bandits
    bandits = [BanditModel(p) for p in BANDITS_PROB]
    
    for n in range(EPISODES):

        #Epsilon greedy
        if rand.random() < EPSILON:

            Bandit = rand.randint(0,2)
            num_times_explored = num_times_explored + 1
            

        else:

            Bandit = numpy.argmax([i.estimatedReward for i in (bandits)])
            num_times_exploited = num_times_exploited + 1

        #Get Reward
        Reward = bandits[Bandit].PullFunc()

        #Update Model with Reward
        bandits[Bandit].UpdateModel(Reward)

        #Update Reward array
        rewards[n] = Reward

    print("TotalReward: ", rewards.sum())
    print("Overrall Win Rate: ", rewards.sum()/EPISODES)
    print("Times Explored: ", num_times_explored)
    print("Times Exploited: ", num_times_exploited)

    cumulative_reward = numpy.cumsum(rewards)
    win_rates = cumulative_reward / (numpy.arange(EPISODES) + 1)
    plt.plot(win_rates)
    plt.plot(numpy.ones(EPISODES) * numpy.max(BANDITS_PROB))
    plt.show()
    


    #for i in range(len(bandits)):
        
    #    print("Reward Bandit[",i,"]: ", bandits[i].estimatedReward)

    return 0

Experiment()
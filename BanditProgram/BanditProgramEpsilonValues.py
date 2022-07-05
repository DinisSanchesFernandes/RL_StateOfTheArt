import random as rand
import numpy
import matplotlib.pyplot as plt

BANDITS_MEAN = [2, 4, 6]
BANDITS_DEVIATION = [1, 1 , 1]

TOTAL_EPISODES = 100000
EPSILONS = [0.1,0.5,0.3]

class BanditModel():

    def __init__(self, Mean, Deviation):

        #Media
        self.Mean = Mean

        #Desvio
        self.Deviation = Deviation
  
        #Stores Estimated Reward Mean 
        self.estimatedReward = 0

        #Stores Number of Iterations
        self.N = 0


    def PullFunc(self):

        #PullFunc Return Normal Distributed Reward
        return rand.normalvariate(self.Mean,self.Deviation)

    def UpdateModel(self, X):

        #Increments Number of iterations
        self.N = self.N + 1

        #Calculates Reward Mean
        self.estimatedReward = ((self.N - 1)* self.estimatedReward + X) * (1 / self.N)

def Experiment(Mean1, Mean2, Mean3, Deviation1, Deviation2, Deviation3, EPSILON):

    #Define Bandits

    Bandits = [BanditModel(Mean1,Deviation1), BanditModel(Mean2, Deviation2), BanditModel(Mean3, Deviation3)]

    print("Mean Bandit 0: ", Bandits[0].Mean)
    print("Mean Bandit 0: ", Bandits[0].Deviation)


    print("Mean Bandit 1: ", Bandits[1].Mean)
    print("Mean Bandit 1: ", Bandits[1].Deviation)


    print("Mean Bandit 2: ", Bandits[2].Mean)
    print("Mean Bandit 2: ", Bandits[2].Deviation)

    RewardList = numpy.zeros(TOTAL_EPISODES)
    num_times_explored = 0
    num_times_exploited = 0

    for EPISODE in range(TOTAL_EPISODES):

        #Epsilon greedy
        if rand.random() < EPSILON:

            #Choose Random Bandit
            Bandit = rand.randint(0,2)
            num_times_explored = num_times_explored + 1
            

        else:

            #Choose Bandit Using Model
            Bandit = numpy.argmax([i.estimatedReward for i in (Bandits)])
            num_times_exploited = num_times_exploited + 1

        #Get Reward
        Reward = Bandits[Bandit].PullFunc()

        #Register Reward
        RewardList[EPISODE] = Reward

        #Update Model
        Bandits[Bandit].UpdateModel(Reward)

    cumulative_average = numpy.cumsum(RewardList) / (numpy.arange(TOTAL_EPISODES) + 1)

    plt.plot(cumulative_average)
    plt.plot(numpy.ones(TOTAL_EPISODES) * Mean1)
    plt.plot(numpy.ones(TOTAL_EPISODES) * Mean2)
    plt.plot(numpy.ones(TOTAL_EPISODES) * Mean3)
    plt.xscale('log')
    plt.show()

    return cumulative_average



cumulative_average_experimment1 = Experiment(2 , 5 , 7 , 1 , 1 , 1 , 0.1)
cumulative_average_experimment2 = Experiment(2 , 5 , 7 , 1 , 1 , 1 , 0.05)
cumulative_average_experimment3 = Experiment(2 , 5 , 7 , 1 , 1 , 1 , 0.01)

plt.plot(cumulative_average_experimment1, label = 'eps = 0.1')
plt.plot(cumulative_average_experimment2, label = 'eps = 0.05')
plt.plot(cumulative_average_experimment3, label = 'eps = 0.01')
plt.plot(numpy.ones(TOTAL_EPISODES) * 2)
plt.plot(numpy.ones(TOTAL_EPISODES) * 5)
plt.plot(numpy.ones(TOTAL_EPISODES) * 7)

plt.legend()
plt.xscale('log')
plt.show()

plt.plot(cumulative_average_experimment1, label = 'eps = 0.1')
plt.plot(cumulative_average_experimment2, label = 'eps = 0.0.05')
plt.plot(cumulative_average_experimment3, label = 'eps = 0.01')
plt.legend()
plt.show()


#Plot Some Shit

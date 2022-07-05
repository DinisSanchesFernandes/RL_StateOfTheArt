import random as rand
import numpy
from numpy.core.fromnumeric import argmax
import matplotlib.pyplot as plt





BANDITS_PROB = [0.2,0.5,0.75]
TOTAL_EPISODES = 10000
EPISODES = 1.

class BanditModel():

    def __init__(self,p):

        #Stores The Probabilitie Of Respective Bandit
        self.p = p

        #Stores Estimated Reward For Respective Bandit
        self.estimatedReward = 0.

        #Stores NUmber of Times Bandit Was Selcted
        self.N = 0.


    def PullFunc(self):

        #Selects Random Number, If Smaller Than the Probabilitie Return 1
        return rand.random() < self.p

    def UpdateModel(self, X):

        #Update Number Of Times Bandit Selected
        self.N = self.N + 1.
        
        #Update Estimate
        self.estimatedReward = ((self.N - 1)* self.estimatedReward + X) * (1 / self.N)



def Experiment():

    #Init Bandits
    Bandits = [BanditModel(p) for p in BANDITS_PROB]

    #Init RewardList
    RewardLists = numpy.zeros(TOTAL_EPISODES)

    EPISODES = 0

    #Select All The Bandits so We dont Divide by zero in the UCB step
    for i in range(len(Bandits)):

        #PullFunc
        Reward = Bandits[i].PullFunc()

        EPISODES += 1

        #UpdateModel
        Bandits[i].UpdateModel(Reward)

        #UpdateModel
        RewardLists[EPISODES] = Reward

    
    #Ciclo
    for EPISODES in range(EPISODES,TOTAL_EPISODES):

        #Choose UCB Step
        #Bandit argmax(estimatedReward + sqrt((2 * N) / Bandits.N))
        Bandit = argmax([(p.estimatedReward + numpy.sqrt((2 * numpy.log10(EPISODES)) / p.N)) for p in Bandits])

        print("Bandits: ", Bandit)

        #PullFunc
        Reward = Bandits[Bandit].PullFunc()
        
        #SaveReward
        RewardLists[EPISODES] = Reward

        #UpdateModel
        Bandits[Bandit].UpdateModel(Reward)

    RewardCumulative = numpy.cumsum(RewardLists)

    print("Number of Time Each Bandit was selected: ", [p.N for p in Bandits])
    print("Estimated Mean for Each Bandit: ", [p.estimatedReward for p in Bandits])
    print("Total Reward: ", numpy.sum(RewardLists))
    print("Win Rate: ", numpy.sum(RewardLists)/TOTAL_EPISODES)

    WinRate = RewardCumulative / (numpy.arange(TOTAL_EPISODES) + 1)
    plt.plot(WinRate)
    plt.plot(numpy.ones(TOTAL_EPISODES) * numpy.max(BANDITS_PROB))
    plt.ylim([0,1])
   #plt.xscale('log')
    plt.show()

Experiment()
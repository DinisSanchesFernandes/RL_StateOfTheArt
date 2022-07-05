import random as rand
import numpy
import matplotlib.pyplot as plt


BANDITS_PROB = [0.2,0.5,0.75]
TOTAL_EPISODES = 10000
EPSILON = 0.1

class BanditModel():

    def __init__(self,p):

        #Stores The Probabilitie Of Respective Bandit
        self.p = p

        #Stores Estimated Reward For Respective Bandit
        self.estimatedReward = 5.

        #Stores NUmber of Times Bandit Was Selcted
        self.N = 1.


    def PullFunc(self):

        #Selects Random Number, If Smaller Than the Probabilitie Return 1
        return rand.random() < self.p

    def UpdateModel(self, X):
        
        #Update Number Of Times Bandit Selected
        self.N = self.N + 1.

        #Update Estimate
        self.estimatedReward = ((self.N - 1)* self.estimatedReward + X) * (1 / self.N)

def Experiment():

    #Init Of Bandits With Repctive Probabilites
    Bandits = [BanditModel(p) for p in BANDITS_PROB]

    #Init Reward List Array, Stores Reward for each Iteration
    RewardList = numpy.zeros(TOTAL_EPISODES)

    for EPISODE in range (TOTAL_EPISODES):

        #Select Best Bandit
        BanditChoice = numpy.argmax([p.estimatedReward for p in Bandits])

        #Reward Stores The Vaule Returned
        Reward = Bandits[BanditChoice].PullFunc()

        #Register the Reward
        RewardList[EPISODE] = Reward

        #Update Function Accodingly with the Reward
        Bandits[BanditChoice].UpdateModel(Reward)

    #Plot

    RewardCumulative = numpy.cumsum(RewardList)

    print("Number of Time Each Bandit was selected: ", [p.N for p in Bandits])
    print("Estimated Mean for Each Bandit: ", [p.estimatedReward for p in Bandits])
    print("Total Reward: ", numpy.sum(RewardList))
    print("Win Rate: ", numpy.sum(RewardList)/TOTAL_EPISODES)



    WinRate = RewardCumulative / (numpy.arange(TOTAL_EPISODES) + 1)
    plt.plot(WinRate)
    plt.plot(numpy.ones(TOTAL_EPISODES) * numpy.max(BANDITS_PROB))
    plt.ylim([0,1])
   #plt.xscale('log')
    plt.show()


Experiment()
import random as rand
from scipy.stats import beta
import numpy
import matplotlib.pyplot as plt

BANDITS_PROB = [0.2,0.5,0.75]
TOTAL_EPISODES = 100000
PLOT_EPISODES = [5, 1000, 50000, 99999]
class BanditModel():

    def __init__(self,p):

        #Stores The Probabilitie Of Respective Bandit
        self.p = p

        #Stores Estimated Reward For Respective Bandit
        self.estimatedReward = 5.

        #Stores NUmber of Times Bandit Was Selcted
        self.N = 1.

        #Prior
        #Alpha
        self.alpha = 1
        #Beta
        self.beta = 1


    def PullFunc(self):

        #Selects Random Number, If Smaller Than the Probabilitie Return 1
        return rand.random() < self.p

    def UpdateModel(self, X):
        
        #Update Number Of Times Bandit Selected
        self.N = self.N + 1.

        #Update Model is done using Bayesian methods
        
        #Prior = Beta(alpha , beta)
        #Posterior = Beta(alpha + X , N + beta - X)
        #alpha = 1
        #beta = 1
        #N = 1

        self.alpha = self.alpha + X
        self.beta = self.beta - X + 1 

    def TakeSample(self):

        return numpy.random.beta(self.alpha,self.beta)
    
def PlotFunction(Bandits, Episode):

    x = numpy.linspace(0,1,200)

    for b in Bandits:

        y = beta.pdf(x, b.alpha, b.beta)
        plt.plot(x,y,label=f"Real Mean:  {b.p:.4f}, win rate = {b.alpha - 1}/{b.N}")
    plt.title(f"Trial: {Episode}")
    plt.legend()
    plt.show()

def Experiment():

    #Init Bandits
    Bandits = [BanditModel(p) for p in BANDITS_PROB]

    print(Bandits[2].alpha)

    #Init Reward List
    RewardList = numpy.zeros(TOTAL_EPISODES)
    
    #Ciclo
    for EPISODE in range(TOTAL_EPISODES):

        #Select Bandit
        Bandit = numpy.argmax([b.TakeSample() for b in Bandits])
        
        #PullFunc
        Reward = Bandits[Bandit].PullFunc()

        RewardList[EPISODE] = Reward

        #UpdateModel
        Bandits[Bandit].UpdateModel(Reward)
        
        if EPISODE in PLOT_EPISODES :

            PlotFunction(Bandits,EPISODE)
            
            print("Optimal Plays: ", Bandits[2].N)
            print("SubOptimal Plays", Bandits[0].N + Bandits[1].N)
            print("Total Reward: ", RewardList.sum())


            

Experiment()
    
    
from numpy.random.mtrand import normal
from scipy.stats import norm
import numpy
import matplotlib.pyplot as plt

BANDITS_PROB = [1, 2, 3]
TOTAL_EPISODES = 100000
PLOT_EPISODES = [5, 1000, 50000, 99999]
class BanditModel():

    def __init__(self,TrueMean):

        #Stores The Probabilitie Of Respective Bandit
        self.TrueMean = TrueMean

        #Stores Estimated Reward For Respective Bandit
        self.estimatedReward = 5.

        #Stores NUmber of Times Bandit Was Selcted
        self.N = 1.

        #Prior = N(Mean,Lambda)
        #Mean = 0
        #Lambda = 1
        self.Xsum = 0
        self.Lambda = 1
        self.Mean = 0
        self.Tau = 1


    def PullFunc(self):

        #A função randn() retorna um valor aleatorio com distribuição normal
        #O Desvio é igual a 1/Precisão 
        #Sendo assim a distribuição Normal de cada bandido será dada por: (X * Desvio²) + Mean = (X / sqrt(Tau)) + Mean
        return numpy.random.randn() / numpy.sqrt( self.Tau ) + self.TrueMean

    def UpdateModel(self, X):
        
        #Update Number Of Times Bandit Selected
        self.N = self.N + 1.

        #Calculate Posterior Parameters

        #X sum 
        #Xsum += X 
        self.Xsum += X

        #Lambda
        #Lambda = Tau * 1 + Lambda
        self.Lambda += self.Tau 

        #Mean
        #Mean = (1 / (Tau*1 + Lambda)) * (Tal * Xsum + Lambda * mean)
        self.Mean = (self.Xsum * self.Tau) * ( 1 /self.Lambda ) 

    def TakeSample(self):

        return numpy.random.randn() / numpy.sqrt( self.Lambda ) + self.Mean
    
def PlotFunction(Bandits, Episode):

    x = numpy.linspace(0,5,1000)

    for b in Bandits:

        y = norm.pdf(x, b.Mean, numpy.sqrt(1/b.Lambda))
        plt.plot(x,y,label=f"Real Mean:  {b.TrueMean:.4f}, Number of Times Played = {b.N}")
    
    plt.title(f"Trial: {Episode}")
    plt.legend()
    plt.show()

def Experiment():

    #Init Bandits
    Bandits = [BanditModel(p) for p in BANDITS_PROB]

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

    plt.plot(Bandits[0].Mean * numpy.ones(TOTAL_EPISODES))
    plt.plot(Bandits[1].Mean * numpy.ones(TOTAL_EPISODES))
    plt.plot(Bandits[2].Mean * numpy.ones(TOTAL_EPISODES))
    plt.show()

Experiment()
    
    
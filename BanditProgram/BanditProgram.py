from random import normalvariate
from random import randint
from random import random
import numpy as nump
from numpy.core.fromnumeric import argmax 

max_iterations = 30000

#Define
class SlotMachine():

    def __init__(self, Mean, Deviation):
        #Mean
        self.Mean = Mean

        #Desvio
        self.Deviation = Deviation

        #TotalReward
        self.TotalReward = 0

    
    def SetMeanDeviation(Mean,Deviation):
        Mean= Mean
    

    def Pull(self):

        Reward = normalvariate(self.Mean, self.Deviation)

        self.TotalReward = Reward

        return Reward

class BanditModel:

    def __init__ (self,epsilondecay):

        self.Reward=0

        self.Slots = [SlotMachine(10,5),SlotMachine(8,4),SlotMachine(5,5)]

        self.Epsilon = 1

        self.RewardArray = [0,0,0] 

        self.EpsilonDecay = epsilondecay 


    def ChoiceModel(self):

        return nump.argmax([self.Slots[0].TotalReward, self.Slots[1].TotalReward, self.Slots[2].TotalReward])


    def EpsilonDecayFunc(self):

        self.Epsilon = self.Epsilon * self.EpsilonDecay

    def MakeChoice(self):

        aux = random()

        self.EpsilonDecayFunc()

        if aux < self.Epsilon:

            return randint(0,2)

        else:

            return self.ChoiceModel()

    def BanditStep(self):

        
        self.Reward = self.Reward + self.Slots[self.MakeChoice()].Pull()




#Main
Bandit1 = BanditModel((2*0.01)/max_iterations)

Bandit2 = BanditModel(1)

#Bandit Random





for t in range(max_iterations):
    
    #BanditMakeChoice
    Bandit1.BanditStep()
    Bandit2.BanditStep()


    #GetReward

    #UpdateModel

print("Bandit1 Reward: ", Bandit1.Reward)
print("Bandit 1 Best Slot: ", nump.argmax([Bandit1.Slots[0].TotalReward, Bandit1.Slots[1].TotalReward, Bandit1.Slots[2].TotalReward]))
print("Bandit2 Reward: ", Bandit2.Reward)

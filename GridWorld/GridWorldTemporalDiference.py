import numpy

from cmath import inf

import matplotlib.pyplot as plt

class GridWorld:

    def __init__(self,rows,cols,start):

        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self,rewardsbuffer,TransitionProbsBuffer):
        #rewards = dict of: (i,j): r(row,cols): reward
        #actions = dict of: (i,j): A (row,col): action list

        self.reward = rewardsbuffer
        self.TransitionProbs = TransitionProbsBuffer

    def set_state(self,s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i,self.j)

    def is_terminal(self,s):
        return s in self.reward
    
    def move(self,action):

        s = (self.i,self.j)
        #print(s)
        next_state_probs = self.TransitionProbs.get((s,action),0)

        #print("State ",s)
        #print("Action ",action)    
        #print(self.TransitionProbs.get((s,action),0))    
        next_states = list(next_state_probs.keys())
        next_probs = list(next_state_probs.values())
        
        s2_index = numpy.random.choice(len(next_states),p=next_probs)

        s2 = next_states[s2_index]

        self.i ,self.j = s2

    def GameOver(self):

        if (self.i,self.j) in self.reward:

            return 1



#Print Function
def print_values(G,V):

    for Row in range(G.rows):

        print("----------------")

        for Col in range(G.cols):

            v = V.get((Row,Col),0)

            if v >= 0:

                print(" %.2f|" % v, end="")

            else:

                print("%.2f|" % v, end="")

        print("")

def print_policy(G,Pol):

    for Row in range(G.rows):

        print("----------------------------------------------------------------------------------------------")

        for Cols in range(G.cols):

            p = Pol.get((Row,Cols),' ')
            print("     %s      |" % p, end="")

        print("")

def Epsilon_Greedy(Pol,Epsilon,s,Actions):

    if numpy.random.sample() < (1-Epsilon):

        return Pol[s]

    else:

        return numpy.random.choice(Actions)



#AllStates(State)
#Array with all states of environment
AllStates = [
    (0,0),
    (0,1),
    (0,2),
    (0,3),
    (1,0),
    (1,2),
    (1,3),
    (2,0),
    (2,1),
    (2,2),
    (2,3)
]

#RewardsBuffer((State) : RewardValue)
#Reward Received in current state
RewardsBuffer = {
    (0,3) : 1 , 
    (1,3) : -1
}
Rewards={}


TransitionProbsBuffer = {
    ((2, 0), 'U'): {(1, 0): 1.0},
    ((2, 0), 'D'): {(2, 0): 1.0},
    ((2, 0), 'L'): {(2, 0): 1.0},
    ((2, 0), 'R'): {(2, 1): 1.0},
    ((1, 0), 'U'): {(0, 0): 1.0},
    ((1, 0), 'D'): {(2, 0): 1.0},
    ((1, 0), 'L'): {(1, 0): 1.0},
    ((1, 0), 'R'): {(1, 0): 1.0},
    ((0, 0), 'U'): {(0, 0): 1.0},
    ((0, 0), 'D'): {(1, 0): 1.0},
    ((0, 0), 'L'): {(0, 0): 1.0},
    ((0, 0), 'R'): {(0, 1): 1.0},
    ((0, 1), 'U'): {(0, 1): 1.0},
    ((0, 1), 'D'): {(0, 1): 1.0},
    ((0, 1), 'L'): {(0, 0): 1.0},
    ((0, 1), 'R'): {(0, 2): 1.0},
    ((0, 2), 'U'): {(0, 2): 1.0},
    ((0, 2), 'D'): {(1, 2): 1.0},
    ((0, 2), 'L'): {(0, 1): 1.0},
    ((0, 2), 'R'): {(0, 3): 1.0},
    ((2, 1), 'U'): {(2, 1): 1.0},
    ((2, 1), 'D'): {(2, 1): 1.0},
    ((2, 1), 'L'): {(2, 0): 1.0},
    ((2, 1), 'R'): {(2, 2): 1.0},
    ((2, 2), 'U'): {(1, 2): 1.0},
    ((2, 2), 'D'): {(2, 2): 1.0},
    ((2, 2), 'L'): {(2, 1): 1.0},
    ((2, 2), 'R'): {(2, 3): 1.0},
    ((2, 3), 'U'): {(1, 3): 1.0},
    ((2, 3), 'D'): {(2, 3): 1.0},
    ((2, 3), 'L'): {(2, 2): 1.0},
    ((2, 3), 'R'): {(2, 3): 1.0},
    ((1, 2), 'U'): {(0, 2): 0.5, (1, 3): 0.5},
    ((1, 2), 'D'): {(2, 2): 1.0},
    ((1, 2), 'L'): {(1, 2): 1.0},
    ((1, 2), 'R'): {(1, 3): 1.0},
}

#Action Space Array
ActionSpace = [
    'U',
    'D',
    'R',
    'L'
]

Policy = {(0,0) : ('R'),
(0,1) : ('R'),
(0,2) : ('R'),
(1,0) : ('U'),
(1,2) : ('R'),
(2,0) : ('U'),
(2,1) : ('R'),
(2,2) : ('U'),
(2,3) : ('U')}

#Hyperparameters
GAMMA = 0.9
ALPHA = 0.1 
MIN_DELTA = 0.001
MAX_EPISODES = 10000
MAX_ITERATION = 20
START_STATE = (2,0)
EPSILON = 0.3

#Init Grid World Object
Grid = GridWorld(3,4,(2,0))
Grid.set(RewardsBuffer,TransitionProbsBuffer)


V = {}

for s in AllStates:

    V[s] = 0

Delta_Array = []

if __name__ == '__main__':


    for _ in range(MAX_EPISODES):

        #Env Reset
        Grid.set_state(START_STATE)
        s = START_STATE

        DELTA = 0

        while not Grid.is_terminal(s):

            s = Grid.current_state()

            a = Epsilon_Greedy(Policy,EPSILON,s,ActionSpace)

            Grid.move(a)

            s1 = Grid.current_state()

            r = Grid.reward.get(Grid.current_state(),0)

            v_old = V[s]

            V[s] = V[s] + ALPHA*(r + GAMMA * V[s1] - V[s])

            DELTA = max(DELTA, numpy.abs(v_old - V[s]))

            #print("State: ",s)
            #print("Prime State: ",s1)
            #print("Reward: ",r)
            #print("DELTA: ",DELTA)
            #print()

            s = s1

        Delta_Array.append(DELTA)

        #if DELTA < MIN_DELTA:

        #    break

    print_values(Grid,V)

    plt.plot(Delta_Array)
    plt.show()        


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

#Best Action
def Best_Action(qtable):

        actionslist = list(qtable.keys())
        valueslist = list(qtable.values())

        maxactionindex = numpy.argmax(valueslist)

        return actionslist[maxactionindex]


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

def Epsilon_Greedy(qtable,epsilon,actionspace):

    if numpy.random.sample() < (1-epsilon):

        return Best_Action(qtable)

    else:

        return numpy.random.choice(actionspace)



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
    ((1, 2), 'U'): {(0, 2): 1.0},
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
EPSILON = 0.1
STEP_COST = -0.1

#Init Grid World Object
Grid = GridWorld(3,4,(2,0))
Grid.set(RewardsBuffer,TransitionProbsBuffer)


Q = {}
for s in AllStates:

    Q[s] = {}

    if not Grid.is_terminal(s):

        for a in ActionSpace:

            Q[s][a] = numpy.random.sample()

    else:

        for a in ActionSpace:

            Q[s][a] = 0

V = {}

for s in AllStates:

    actionslist = list(Q[s].keys())
    valueslist = list(Q[s].values())

    maxactionindex = numpy.argmax(valueslist)

    V[s] = valueslist[maxactionindex]       

print_values(Grid,V)

Delta_Array = []
Reward_Array = []

if __name__ == '__main__':

    #For range(MAX_ITERATION)
    for EPISODE in range(MAX_EPISODES): 

        RewardEpisode = 0

        #Set State to START_STATE
        Grid.set_state(START_STATE)
        s = START_STATE

        #While State0 not terminal 
        while not Grid.is_terminal(s):

            #Get Action0 Epsilon Greedy(State0)
            a = Epsilon_Greedy(Q[s],EPSILON,ActionSpace)

            #State1 = Move(Action0)
            Grid.move(a)
            s1 = Grid.current_state()
            
            #Reward = Get Reward
            r = Grid.reward.get(s1,STEP_COST)

            RewardEpisode += r

            a1 = Best_Action(Q[s1])

            #Q[State0][Action0] = Q[State0][Action0] + ALPHA(Reward + GAMMA * Q[State1][Action1] - Q[State0][Action0])
            Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * Q[s1][a1] - Q[s][a]) 

            #State0 = State1
            s = s1
            

        Reward_Array.append(RewardEpisode)
    #plt.plot(Delta_Array)
    #plt.show()        

    V = {}
    Policy = {}

    for s in AllStates:

        actionslist = list(Q[s].keys())
        valueslist = list(Q[s].values())

        maxactionindex = numpy.argmax(valueslist)

        V[s] = valueslist[maxactionindex] 
        Policy[s] = actionslist[maxactionindex]

    print_policy(Grid,Policy)
    print_values(Grid,V)

    plt.plot(Reward_Array)
    plt.show()
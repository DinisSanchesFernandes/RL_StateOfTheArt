
from ast import Return
import numpy

class GridWorld:

    def __init__(self,rows,cols,start):

        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self,rewardsbuffer,actionsbuffer,TransitionProbsBuffer):
        #rewards = dict of: (i,j): r(row,cols): reward
        #actions = dict of: (i,j): A (row,col): action list

        self.reward = rewardsbuffer
        self.actions = actionsbuffer
        self.TransitionProbs = TransitionProbsBuffer

    def set_state(self,s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i,self.j)

    def is_terminal(self,s):
        return s not in self.actions

    def get_next_state(self,s,a):
        
        i,j = s[0],s[1]

        if a in self.actions[(i,j)]:

            if a == 'U':
                i -= 1

            elif a == 'D':
                i += 1

            elif a == 'R':
                j += 1

            elif a == 'L':
                j -= 1

        return i,j
    
    def move(self,action):

        s = (self.i,self.j)
        print(s)
        next_state_probs = self.TransitionProbs.get((s,action),0)
        
        next_states = list(next_state_probs.keys())
        next_probs = list(next_state_probs.values())
        
        s2_index = numpy.random.choice(len(next_states),p=next_probs)

        s2 = next_states[s2_index]

        self.i ,self.j = s2

    def all_states(self):

        return set(self.actions.keys()) | set(self.rewards.keys())

    def GameOver(self):

        if (self.i,self.j) in Grid.reward:

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

AllNonTerminalStates= [
    (0,0),
    (0,1),
    (0,2),
    (1,0),
    (1,2),
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


#Deterministic Policie
Policie = {(0,0) : ('R'),
(0,1) : ('R'),
(0,2) : ('R'),
(1,0) : ('U'),
(1,2) : ('R'),
(2,0) : ('U'),
(2,1) : ('R'),
(2,2) : ('U'),
(2,3) : ('U')}

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

#ActionsList((State) : (PossibleActions,(...)))
#All Possible Actions
ActionsBuffer = {
    (0,0) : ('D','R'),
    (0,1) : ('R','L'),
    (0,2) : ('D','R','L'),
    (1,0) : ('D','U'),
    (1,2) : ('D','U','R'),
    (2,0) : ('U','R'),
    (2,1) : ('L','R'),
    (2,2) : ('U','L','R'),
    (2,3) : ('L','U')
}

#Action Space Array
ActionSpace = [
    'U',
    'D',
    'R',
    'L'
]


#Init Value Function
V = {}

#Hyperparameters
ALPHAThreshold = 0.001

#Hyperparameters
GAMMA = 0.9



#Init Grid World Object
Grid = GridWorld(3,4,(2,0))
Grid.set(RewardsBuffer,ActionsBuffer,TransitionProbsBuffer)

TOTAL_EPISODES = 20
TOTAL_ITERATIONS = 5
#START_STATE = (2,0)
EPISODE_ARRAY = {}
STATE_ARRAY = {}
RETURN_MATRIX = {}

#Value Function set zeros
for s in AllStates:
    V[s] = 0
    RETURN_MATRIX[s] = []

for EPISODE in range (TOTAL_EPISODES):

    #Set Random Start State
    START_STATE_index = numpy.random.choice(len(AllNonTerminalStates))

    START_STATE = AllNonTerminalStates[START_STATE_index]

    Grid.set_state(START_STATE)
    
    for ITERATION in range (TOTAL_ITERATIONS):

        s = (Grid.i,Grid.j)

        STATE_ARRAY[ITERATION] = s

        a = Policie.get(s)

        Grid.move(a)

        s1 = (Grid.i,Grid.j)

        r = Grid.reward.get((s1),0)

        EPISODE_ARRAY[ITERATION] = (s,a,s1,r)

        ITERATION += 1

        if (Grid.GameOver()):

            break
    
    G = 0

    while ITERATION > 0:

        ITERATION = ITERATION - 1

        s,a,s1,r = EPISODE_ARRAY[ITERATION] 

        G = r + GAMMA * G

        if s not in list(STATE_ARRAY.values())[0:ITERATION]:

            RETURN_MATRIX[s].append(G)
            V[s] = numpy.mean(RETURN_MATRIX[s])

    print_values(Grid,V)

  

    

    

    

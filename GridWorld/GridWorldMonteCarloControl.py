
import numpy

from cmath import inf




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

    def all_states(self):

        return set(self.actions.keys()) | set(self.rewards.keys())

    def GameOver(self):

        if (self.i,self.j) in self.reward:

            return 1


STATE_ARRAY = []
ACTION_ARRAY = []
REWARD_ARRAY = [0]

def Play_Game(START_STATE,TOTAL_ITERATIONS,Grid):

    Grid.set_state(START_STATE)

    for ITERATION in range (TOTAL_ITERATIONS):

        s = (Grid.i,Grid.j)

        STATE_ARRAY.append(s)

        a = Policy.get(s)

        ACTION_ARRAY.append(a)

        Grid.move(a)

        s1 = (Grid.i,Grid.j)

        r = Grid.reward.get((s1),0)

        REWARD_ARRAY.append(r)

        ITERATION += 1

        if (Grid.GameOver()):

            break
    
    return REWARD_ARRAY, STATE_ARRAY, ACTION_ARRAY



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

def Best_Action(Q_table):

    print("Qtable: ",Q_table.values())
    print()
    

    max_value = max(Q_table.values())

    max_keys = [key for key,val in Q_table.items() if val == max_value]

    return numpy.random.choice(max_keys)



#Init Value Function
V = {}

#Hyperparameters
ALPHAThreshold = 0.001

#Hyperparameters
GAMMA = 0.9



#Init Grid World Object
Grid = GridWorld(3,4,(2,0))
Grid.set(RewardsBuffer,ActionsBuffer,TransitionProbsBuffer)

TOTAL_EPISODES = 100
TOTAL_ITERATIONS = 5
#START_STATE = (2,0)

RETURN_MATRIX = {}

Q = {}


#Value Function set zeros
for s in AllStates:

    if not Grid.is_terminal(s):
    
        Q[s] = {}

        for a in ActionSpace:
            RETURN_MATRIX[s,a] = []
            Q[s][a] = 0
    else:

        pass

Policy = {}
for s in AllStates:
    if not Grid.is_terminal(s):
        Policy[s] = numpy.random.choice(ActionSpace)

print_policy(Grid,Policy)

for EPISODE in range (TOTAL_EPISODES):

    #Set Random Start State
    START_STATE_index = numpy.random.choice(len(AllNonTerminalStates))

    START_STATE = AllNonTerminalStates[START_STATE_index]


    #Input TOTAL_ITERATIONS, START_STATE, Grid
    
    REWARD_ARRAY,STATE_ARRAY,ACTION_ARRAY = Play_Game(START_STATE,TOTAL_ITERATIONS,Grid)

    G = 0

    states_actions = list(zip(STATE_ARRAY,ACTION_ARRAY))

    T = len(STATE_ARRAY)

    print("Evaluate------------------------------------")

    for t in range(T-1,-1,-1):

        t = t - 1

        #s,a,s1,r = EPISODE_ARRAY[ITERATION] 

        s = STATE_ARRAY[t]

        r = REWARD_ARRAY[t + 1]

        a = ACTION_ARRAY[t]

        G = r + GAMMA * G

        if (s,a) not in states_actions[:t]:

            RETURN_MATRIX[s,a].append(G)
            Q[s][a] = numpy.mean(RETURN_MATRIX[s,a])

            
            Besta = Best_Action(Q[s])
            print("BestAction: ", Besta)
            #print("Best Action: ",Besta)
            Policy[s] = Besta

        #Get Best Action
        #print("State: ",s," Best Action :",Best_Action(Q,s))
        

    #print("Q: ",Q)

    #print("Return ",RETURN_MATRIX)

    print_policy(Grid,Policy)

  

    

    

    

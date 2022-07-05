
import numpy

from cmath import inf




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

def Play_Game(StartState,MaxIterations,Grid,Policy,ActionSpace):



    Grid.set_state(StartState)

    s = Grid.current_state()

    a = numpy.random.choice(ActionSpace)

    StateArray = [s]

    ActionArray = [a]

    RewardArray = [0]

    for _ in range (MaxIterations):

        Grid.move(a)

        r = Grid.reward.get(Grid.current_state(),0)

        s = Grid.current_state()

        RewardArray.append(r)

        StateArray.append(s)

        if Grid.GameOver():
            break
        else:
            a = Policy[s]
            ActionArray.append(a)
    
    return RewardArray, StateArray, ActionArray

def Best_Action(Q_list):

    GreatestValue = max(Q_list.values())

    BestActions = [Action for Action,Value in Q_list.items() if Value == GreatestValue]

    return numpy.random.choice(BestActions),GreatestValue

def Random_State(Grid,StateSpace):

    NonTerminalState = [State for State in StateSpace if not Grid.is_terminal(State)]

    return NonTerminalState[numpy.random.choice(len(NonTerminalState))]

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

def print_policy_probabilites(G,Pol):

    for Row in range(G.rows):

        print("----------------------------------------------------------------------------------------------")

        for a in ActionSpace:

            for Col in range(G.cols):

                State = (Row,Col)
                StateDic = Pol.get(State,0)

                if StateDic != 0:

                    p = StateDic.get(a,0)
                    print("     ",a,": %.2f     |" % p, end="")

                else:

                    print("                     |", end="")

            print("")
            
def choose_policy_action(Policy,State):

    Action_Prob = Policy.get(State)

    ActionList = list(Action_Prob.keys())
    ProbList = list(Action_Prob.values())

    ActionIndex = numpy.random.choice(len(ActionList),p=ProbList)

    return ActionList[ActionIndex]




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

#Hyperparameters
GAMMA = 0.9
MAX_EPISODES = 10000
MAX_ITERATION = 20

#Init Grid World Object
Grid = GridWorld(3,4,(2,0))
Grid.set(RewardsBuffer,TransitionProbsBuffer)

#Init Q Table
Q_Table = {}

#Init Number Of Samples Counter
NumberOfSamples = {}

#Init Policy
Policy = {}

#Set All Q_table Values to Zero
#Set All Sample Counter to Zero
for s in AllStates:
    
    if not Grid.is_terminal(s):

        Q_Table[s] = {}
        NumberOfSamples[s] = {}

        for a in ActionSpace:

            NumberOfSamples[s][a] = 0
            Q_Table[s][a] = 0

for s in AllStates:

    Policy[s] = {}

    if not Grid.is_terminal(s):

        for a in ActionSpace:

            RandProbPol = numpy.random.random_sample()

            while RandProbPol == 0:

                RandProbPol = numpy.random.random_sample()

            Policy[s][a] = RandProbPol    


def Monte_Carlo():

    print_policy(Grid,Policy)

    for _ in range(MAX_EPISODES):

        StartState = Random_State(Grid,AllStates)

        RewardArray, StateArray, ActionArray = Play_Game(StartState,MAX_ITERATION,Grid,Policy,ActionSpace)

        G = 0

        T = len(StateArray)

        StateActionPairsArray = list(zip(StateArray,ActionArray))

        #print("Rewards: ",RewardArray)
        #print("States: ",StateArray)
        #print("Actions",ActionArray)

        #T - 2 => Pq a contagem vai até (MaxIterations - 1) isto pq começa em zero a contagem
        #Tendo isto em conta o index do array vai até (MaxIterations - 1)
        #Como Nos não contamos o estado terminal fica (MaxIterations - 2)

        #O Stop é não inclusive ou seja que isto para no zero
        for t in range(T-2,-1,-1):
            
            r = RewardArray[t + 1]

            a = ActionArray[t]

            s = StateArray[t]

            G = r + GAMMA * G

            if not (s,a) in StateActionPairsArray[:t]:

                #print("Firs Time")

                NumberOfSamples[s][a] += 1
                NumberOfSamplesAux = 1/ NumberOfSamples[s][a]
                Old_Q = Q_Table[s][a]

                Q_Table[s][a] = Old_Q + NumberOfSamplesAux*(G - Old_Q)
                Policy[s] = Best_Action(Q_Table[s])[0]

    print()
    print("------------------------------------------------------")
    print_policy(Grid,Policy)

    V = {}
    for s,Qs in Q_Table.items():
        V[s] = Best_Action(Q_Table[s])[1]

    print_values(Grid,V)


if __name__ == '__main__':

    print_policy_probabilites(Grid,Policy)

    choose_policy_action(Policy,(0,0))

from cmath import inf
import numpy

class GridWorld:

    def __init__(self,rows,cols,start):

        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self,actionsbuffer):
        
        self.actions = actionsbuffer
        
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
        next_state_probs = self.TransitionProbs[(s,a)]
        next_states = list(next_state_probs.keys())
        next_probs = list(next_state_probs.values())
        s2 = numpy.random.choice(next_states,p=next_probs)

        self.i ,self.j = s2

    def all_states(self):

        return set(self.actions.keys()) | set(self.rewards.keys())


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


def Standard_GridWorld():

    Grid = GridWorld(3,4,(2,0))

    Actions = {(0,0) : ('D','R'),
    (0,1) : ('R','L'),
    (0,2) : ('D','R','L'),
    (1,0) : ('D','U'),
    (1,2) : ('D','U','R'),
    (2,0) : ('U','R'),
    (2,1) : ('L','R'),
    (2,2) : ('U','L','R'),
    (2,3) : ('L','U')}

    Grid.set(Actions)

    return Grid

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


Policie = {(0,0) : 'R',
(0,1) : 'R',
(0,2) : 'D',
(1,0) : 'U',
(1,2) : 'R',
(2,0) : 'R',
(2,1) : 'R',
(2,2) : 'R',
(2,3) : 'U'}


ActionSpace = ['U','D','L','R']

RewardBuffer = {(0,3): 1, (1,3): -1}
Rewards = {}

TransitionProbs = {}

ALPHAThreshold = 0.001

GAMMA = 0.9

#Init V
V = {}
for s in AllStates:
    V[s] = 0

Grid = Standard_GridWorld()

#Init Reward and TransitionProb 
for Row in range(Grid.rows):
    
    for Col in range(Grid.cols):
    
        s = (Row,Col)

        if not Grid.is_terminal(s):

            for a in ActionSpace:

                s1 = Grid.get_next_state(s,a)

                TransitionProbs[(s,a,s1)] = 1

                if s1 in RewardBuffer:

                    Rewards[(s,a,s1)] = RewardBuffer[s1]

def Policy_Evaluation():

    it = 0
    #print("---------------------Policy Evaluation---------------------")


    while True:

        #print("Iteration ", it)

        MaxVariation = 0
        it += 1

        for s in AllStates:

            if not Grid.is_terminal(s):

                v_old = V[s]
                v_new = 0

                for a in ActionSpace:

                    for s1 in AllStates:

                        r = Rewards.get((s,a,s1),0)

                        TransProb = TransitionProbs.get((s,a,s1),0)

                        if Policie.get(s) == a:

                            ActionProb = 1

                            #if r == 1 or r == -1:
                                #print("------------------------")
                                #print("TransProb: ", TransProb)
                                #print("ActionProb: ", ActionProb)
                                #print("Reward: ", r)
                                #print("V[s1]: ", V[s1])

                        else: 

                            ActionProb = 0

     
                        v_new += ActionProb * TransProb * (r + GAMMA * V[s1])

                V[s] = v_new
                MaxVariation = max(MaxVariation, numpy.abs(v_old-V[s]))



        if MaxVariation < ALPHAThreshold:

            break

def Policy_improvement():

    ActionCounter = 0
    ActionArray = {}

    #print("---------------------Policy Improvement---------------------")

    is_policy_stable = True

    for s in AllStates:

        BestAction = None
        BestValue = float(-inf)

        #print("----------")
        #print("State: ",s)

        if not Grid.is_terminal(s):


            a_old = Policie.get((s))

            for a in ActionSpace:

                value = 0

                for s1 in AllStates:

                    TransProb = TransitionProbs.get((s,a,s1),0)

                    r = Rewards.get((s,a,s1),0)

                    value += TransProb * (r + GAMMA * V[s1])


                if value > BestValue:
                    BestValue = value
                    BestAction = a

            #print("Action: ",BestAction)
            #print("Value: ", BestValue)


            #print("Best Action: ", BestAction)
            
            Policie[s] = BestAction

            if BestAction != a_old:

                is_policy_stable = False



    return is_policy_stable

    

#Policy_Evaluation()

it = 0

print("Iteration: ", it)
print_policy(Grid,Policie)
print_values(Grid,V)


print()
print()
print()
print()



while 1:

    it += 1

    Policy_Evaluation()


    print("Policy Evaluation Iteration: ", it)
    print_policy(Grid,Policie)
    print_values(Grid,V)

    print()
    print()
    print()
    print()


    if Policy_improvement():

        break


    print("Policy Improvement Iteration: ", it)
    print_policy(Grid,Policie)
    print_values(Grid,V)


    print()
    print()
    print()
    print()


print_policy(Grid,Policie)
print_values(Grid,V)
print("Optimal Policy Found")


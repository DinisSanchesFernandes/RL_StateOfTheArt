import numpy
from cmath import inf


class GridWorld:

    def __init__(self,rows,cols,start):

        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self,rewardsbuffer,actionsbuffer,transitionprobbuffer):
        #rewards = dict of: (i,j): r(row,cols): reward
        #actions = dict of: (i,j): A (row,col): action list

        self.reward = rewardsbuffer
        self.actions = actionsbuffer
        self.TransitionProbs = transitionprobbuffer

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
step_cost = -0.5
RewardsBuffer = {
    (0, 0): step_cost,
    (0, 1): step_cost,
    (0, 2): step_cost,
    (1, 0): step_cost,
    (1, 2): step_cost,
    (2, 0): step_cost,
    (2, 1): step_cost,
    (2, 2): step_cost,
    (2, 3): step_cost,
    (0,3) : 1 , 
    (1,3) : -1
}
Rewards={}

#TransitionProbsBuffer((State) : {(State1) : 0.7, (State2) : 0.3})
#Probabilitie of Transition to Each Next Possible State
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

TransitionProbs = {}



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


#Value Function set zeros
for s in AllStates:
    V[s] = 0

#Init Grid World Object
Grid = GridWorld(3,4,(2,0))
Grid.set(RewardsBuffer,ActionsBuffer,TransitionProbsBuffer)

for (s,a) , Prob in Grid.TransitionProbs.items():

    for s1,P in Prob.items():

        Rewards[(s,a,s1)] = Grid.reward.get(s1,0)
        TransitionProbs[(s,a,s1)] = P

Policy = {}
for s in AllStates:

    if not Grid.is_terminal(s):

        Policy[s] = numpy.random.choice(ActionSpace)




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

                        if Policy.get(s) == a:

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

            #print("End")

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


            a_old = Policy.get((s))
            for a in ActionSpace:

                value = 0

                for s1 in AllStates:

                    TransProb = TransitionProbs.get((s,a,s1),0)

                    r = Rewards.get((s,a,s1),0)

                    value += TransProb * (r + GAMMA * V[s1])
                    #print("Reward: ",r)
                    #print("value: ",r)
                    #print("TransProb: ",r)


                if value > BestValue:
                    BestValue = value
                    BestAction = a

            #print("Action: ",BestAction)
            #print("Value: ", BestValue)


            #print("Best Action: ", BestAction)
            
            Policy[s] = BestAction

            if BestAction != a_old:

                is_policy_stable = False



    return is_policy_stable


#MAIN
if __name__ == '__main__':

    it = 0

    while 1:

        Policy_Evaluation()



        if Policy_improvement() == True:

            break

        print("------------------------------------------------------------------------")
        print("Iteration: ",it)
        print_policy(Grid,Policy)
        print_values(Grid,V)


        it += 1
        
    #print("Policies: ",Policie[AllStates[0]].get('U'))


    
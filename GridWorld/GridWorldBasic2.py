#------------
#Dictionaries
#------------

import numpy




TransitionProbs = {} 

actions = {}
RewardList = {}
Rewards = {}

V={}


AllStates = [(0,0),
(0,1),
(0,2),
(0,3),
(1,0),
(1,2),
(1,3),
(2,0),
(2,1),
(2,2),
(2,3)]

Policie = {(0,0) : 'R',
(0,1) : 'R',
(0,2) : 'R',
(1,0) : 'U',
(1,2) : 'U',
(2,0) : 'U',
(2,1) : 'R',
(2,2) : 'U',
(2,3) : 'L'}

#Policie = {(0,0) : 'R',
#(0,1) : 'R',
#(0,2) : 'R',
#(1,0) : 'U',
#(1,2) : 'R',
#(2,0) : 'U',
#(2,1) : 'R',
#(2,2) : 'U',
#(2,3) : 'U'}


ActionSpace = ['U','D','L','R']

threshold = 0.01

Lambda = 0.9

class GridWorld:

    def __init__(self,rows,cols,start):

        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self,rewards,actions):
        #rewards = dict of: (i,j): r(row,cols): reward
        #actions = dict of: (i,j): A (row,col): action list

        self.reward = rewards
        self.actions = actions

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

        if action in self.actions[(self.i,self.j)]:
            
            if action == 'U':
                self.i -= 1

            elif action == 'D':
                self.i += 1

            elif action == 'R':
                self.j += 1

            elif action == 'L':
                self.j -= 1

        return self.reward.get((self.i,self.j),0)

    def all_states(self):

        return set(self.actions.keys()) | set(self.rewards.keys())





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





def print_policy(P,G):

    for Row in range(G.rows):

        print("----------------")

        for Col in range(G.cols):

            a = P.get((Row,Col),' ')

            print(" %s |" % a, end="")
        
        print("")



        




def StandardGrid():

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

    Rewards = {(0,3): 1, (1,3): -1}

    Grid.set(Rewards,Actions)

    return Grid

Grid = StandardGrid()

print(Grid.reward)

for i in range(Grid.rows):

    for j in range(Grid.cols):

        s = (i,j)

        if not Grid.is_terminal(s):

            for a in ActionSpace:

                s1 = Grid.get_next_state(s,a)

                TransitionProbs[(s,a,s1)] = 1

                if s1 in Grid.reward:

                    RewardList[(s,a,s1)] = Grid.reward[s1]

print(RewardList)

#V=0

for s in AllStates:

    V[s] = 0

print_values(Grid,V)

it = 0

ALPHA = 0

while True:

    maxVariation = 0

    it += 1

    for s in AllStates:

        if not Grid.is_terminal(s):

            V_old = V[s]
            V_new = 0

            for a in ActionSpace:

                for s1 in AllStates:

                    if a == Policie.get(s):

                        policievalue = 1

                    else:

                        policievalue = 0
                    

                    r = RewardList.get((s,a,s1),0)

                    V_new += policievalue * TransitionProbs.get((s,a,s1),0) * (r + Lambda * V[s1])


            V[s] = V_new

            maxVariation = max(maxVariation, numpy.abs(V_old-V[s]))

    print("Iteration: ",it)
    print_values(Grid,V)


    if threshold > maxVariation:

       break
    





#print(Rewards)
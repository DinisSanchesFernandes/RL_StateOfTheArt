import numpy

from cmath import inf

import matplotlib.pyplot as plt

from sklearn.kernel_approximation import RBFSampler


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


def gather_samples(grid, n_episodes=10000):

    samples = []

    for _ in range(n_episodes):
    
        grid.set_state((2,0))
        s = grid.current_state()
        samples.append(s)
    
        while not grid.GameOver():
    
            a = numpy.random.choice(ActionSpace)
            grid.move(a)
            s = grid.current_state()
            samples.append(s)
    
    return samples
        

class RBFModel:

    def __init__(self,grid):

        #Create RBF Object
        self.FeatureObject = RBFSampler()

        #Gather Trainign data
        samples = gather_samples(grid)

        #Train RBF Object com samples
        self.FeatureObject.fit(samples)

        #Get Dimensions of Phi Object
        dims = self.FeatureObject.random_offset_.shape[0]

        print(dims)

        self.w = numpy.zeros(dims)

        print(self.w)

    #Get Model Values 
    def predict(self,s):

        x = self.FeatureObject.transform([s])[0]

        return x @ self.w 

    
    def grad(self,s):

        x = self.FeatureObject.transform([s])[0]

        return x 





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
ALPHA = 0.01 
MIN_DELTA = 0.001
MAX_EPISODES = 20000
MAX_ITERATION = 20
START_STATE = (2,0)
EPSILON = 0.1

#Init Grid World Object
Grid = GridWorld(3,4,(2,0))
Grid.set(RewardsBuffer,TransitionProbsBuffer)

#Init Rbf Model Class Object
Phi = RBFModel(Grid)

ErrorArray = []


if __name__ == '__main__':

    for _ in range(MAX_EPISODES):

        #Env Reset
        Grid.set_state(START_STATE)
        s = START_STATE
        Vs = Phi.predict(s)

        EpisodeError = 0
        nsteps = 0

        while not Grid.GameOver():

            s = Grid.current_state()

            a = Epsilon_Greedy(Policy,EPSILON,s,ActionSpace)

            Grid.move(a)

            nsteps += 1

            s1 = Grid.current_state()

            r = Grid.reward.get(s1,0)

            if Grid.is_terminal(s1):

                y = r
            
            else:

                Vs2 = Phi.predict(s1)
                y = r + GAMMA * Vs2

            g = Phi.grad(s)
            err = y - Vs
            Phi.w += ALPHA * err * g 

            EpisodeError += err*err
            
            s = s1
            Vs = Vs2

        mse = EpisodeError/nsteps
        ErrorArray.append(mse)

    #print(ErrorArray)
    plt.plot(ErrorArray)
    plt.show()

    V = {}

    for s in AllStates:
        V[s] = Phi.predict(s)

    print_values(Grid,V)

    #for s in AllStates:
    #    print(Phi.grad(s))
    #print(Phi.w)




import numpy as nump

TerminalState = {(0,3)}
Rewards = {(0,3) : 1, 
(1,3): -1,
}

Actions = {(0,0) : ('D','R'),
(0,1) : ('R','L'),
(0,2) : ('D','R','L'),
(1,0) : ('D','U'),
(1,2) : ('D','U','R'),
(2,0) : ('U','R'),
(2,1) : ('L','R'),
(2,2) : ('U','L','R'),
(2,3) : ('L','U')}

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

Policie = {(0,0) : ('R'),
(0,1) : ('R'),
(0,2) : ('R'),
(1,0) : ('U'),
(1,2) : ('R'),
(2,0) : ('U'),
(2,1) : ('R'),
(2,2) : ('U'),
(2,3) : ('U')}

EnvironmentProbabilitiesDic = {}
RewardDic =  {}

MAX_ALPHA = 0.001
GAMMA = 1
TOTAL_EPISODES = 1
TOTAL_ITERATIONS = 9
START_STATE =[2,0]


ACTION_SPACE = ['U','D','R','L']

class GridWorld():

    #Init (Rows, Cols, StartPosition)
    def __init__ (self, Rows, Cols, StartPosition):

        #Init Rows Variable
        self.rows = Rows

        #Init Columns Variable
        self.cols = Cols

        #Init StartPosition Structure X
        self.stateX = StartPosition[0]

        #Init StartPosition Structure Y
        self.stateY = StartPosition[1]

        self.ValueFunction = 0

    #Return True If Terminal state
    def IsTerminal(self,state):

        if state not in Actions:
            return 1
        else:
            return 0
 
    def EnvironmentProbabilities(self):

        #Cycle al the states
        for PresentState in AllStates: 

            #Set State
            #self.set_state(PresentState)

            print("state: ",PresentState)

            #For each state cycle all the possible actions
            if not self.IsTerminal(PresentState):

                for a in Actions[PresentState]:

                    FutureState = self.get_next_state(PresentState,a)

                    EnvironmentProbabilitiesDic[PresentState, FutureState, a] = 1         

                    if FutureState in Rewards:

                        RewardDic[(PresentState,FutureState, a)] = Rewards[FutureState]
                   
                    else: 

                        RewardDic[(PresentState,FutureState, a)] = -1



    #Update State
    def set_state(self, NewState):

        #Change State Data Strucuture
        self.stateX = NewState[0]
        self.stateY = NewState[1]
        

    #Return CurrentState
    def current_state(self):

        return (self.stateX,self.stateY)

    #GetNextState
    def get_next_state(self,state,action):

        i,j = state[0], state[1]

        if action in Actions[i,j]:

            if action == 'U':

                i -= 1

            if action == 'D':

                i += 1

            if action == 'L':

                j -= 1

            if action == 'R':

                j += 1
        
        #Return Array [NextX, NextY]
        return i,j

    def move (self, action):

        #ChangeCurrentState

        if action in Actions[self.stateX,self.stateY]:

            if action == 'U':

                self.stateX -= 1

            if action == 'D':

                self.stateX += 1

            if action == 'L':

                self.stateY -= 1

            if action == 'R':

                self.stateY += 1
        
        return Rewards.get((self.stateX,self.stateY))

    def all_states(self):

        return 
        
    def undo_move(self,action):

        if action == 'U':

            self.stateX -= 1

        elif action == 'D':

            self.stateX += 1

        elif action == 'L':

            self.stateY -= 1

        elif action == 'R':

            self.stateY += 1

        assert (self.current_state() in AllStates)
            #Return Reward



#------
#-MAIN-
#------

V= {}

for saux in AllStates:
    V[saux] = 0

AlphaThreshold = 0.01
Grid = GridWorld(3,4,[2,0])

MaxAlphaDeviation = 1

Grid.EnvironmentProbabilities()

while True:

    MaxAlphaDeviation = 0

    #Cycle To Pass Through All States
    #vk+1(s)=∑aπ(a|s)∑s′∑rp(s′,r|s,a)[r+γ∗Vk(s′)]

    for s in AllStates:

        #If State Not Terminal
        if not Grid.IsTerminal(s):

            #Prepare Value Function Array
            #For Each State This Array Stores a Value
            #Store in Variable The Old Value To then Calculate Alpha
            V_old = V[s]
            new_v = 0

            #∑a
            for a in ACTION_SPACE:

                #∑s′
                for s1 in Actions[s]:

                    #π(a|s)
                    #The Policy is Deterministic
                    Policie_s1 = Grid.get_next_state(s,a)
                    if Policie[s] == a:

                        #EnvironmentProbabilitiesDic[s, s1, a] = 1
                        ActionProb = 1 

                    else:

                        #EnvironmentProbabilitiesDic[s, s1, a] = 0
                        ActionProb = 0

                    r = Rewards.get((s,s1,a),0)

                    #∑p(s′,r|s,a)[r+γ∗Vk(s′)]
                    new_v += (r + GAMMA*V[s1]) * EnvironmentProbabilitiesDic.get((s, s1, a),0) * ActionProb

            V[s] = new_v

            #Δ=maxs|vk+1(s)−vk(s)|
            MaxAlphaDeviation = max(MaxAlphaDeviation, nump.abs(V[s]-V_old))

        

    if MaxAlphaDeviation < AlphaThreshold:

        break

    print(Grid.ValueFunction)


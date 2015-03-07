import numpy as np
import math
import random

class MDP(object):
    """Markov Decision Process (MDP) class.

    THIS IS NOT A GENERAL MDP SOLVER.

    MDP is build for gridworlds where agents can travel horizontally, vertically, and diagonally.

    Attributes:
    use self.Display() to see all attributes
        S      A list of states
        A      A list of actions
        T      Transition matrix T[S,A,S] (StartingState,Action,TargetState)
        R      Reward function
        diagonal boolean variable indicating if agent can travel diagonally
        gamma  Future discount parameter
    """

    def __init__(self, S=[], A=[], T=[], R=[[],[]], diagonal=True, gamma=0.1):
        self.S = S
        self.A = A
        self.T = T
        # This is a simplified version of creating R[A,S]->R
        # Because horizontal and vertical actions have identical costs, and diagonal actions
        # have identical costs (given the state), we can represent the full reward function
        # with two mappings
        self.Rstraight = R[0]
        self.Rdiagonal = R[1]
        self.gamma = gamma
        self.diagonal = diagonal
        self.vals = np.zeros((1, len(S)))
        # Where we'll store the softmaxed probabilities
        self.policy = np.zeros((len(A), len(S)))

    def ValueIteration(self, epsilon=0.0000001):
        self.vals = np.zeros(self.vals.shape)
        while True:
            V2 = self.vals.copy()
            for i in range(0, len(self.S)):
                # horizontal and vertical actions are always 0 to 3, and diagonal actions are always 4 to 7
                # Update value iteration depending on the action taken.
                StraightValues = (self.Rstraight[i] + self.gamma * (np.mat(self.T[i, range(0,4),:]) * np.mat(V2.transpose()))).max()
                if self.diagonal:
                    DiagonalValues = (self.Rdiagonal[i] + self.gamma * (np.mat(self.T[i, range(4,8),:]) * np.mat(V2.transpose()))).max()
                    self.vals[0, i] = max(StraightValues,DiagonalValues)
                else:
                    self.vals[0, i] = StraightValues
            if (self.vals - V2).max() <= epsilon:
                break

    def BuildPolicy(self, tau=0.01, Softmax=True):
        # Build a policy using the results from value iteration
        for i in range(0, len(self.S)):
            optionsstraight = np.mat(self.T[i, range(0,4),:]) * np.mat(self.vals.transpose()) + self.Rstraight[i]
            if self.diagonal:
                optionsdiagonal = np.mat(self.T[i, range(4,8),:]) * np.mat(self.vals.transpose()) + self.Rdiagonal[i]
                options = np.concatenate([optionsstraight,optionsdiagonal])
            else:
                options = np.concatenate([optionsstraight])
            # Prevent softmax from overflowing
            options = options-abs(max(options))
            # Softmax the policy
            if Softmax:
                try:
                    options = [math.exp(options[j] / tau)
                               for j in range(len(options))]
                except OverflowError:
                    print "WARNING: Failed to softmax policy"
                    raise
                # If all actions have no value then set a uniform distribution
                if sum(options)==0:
                    self.policy[:,i] = [1.0/len(options) for j in range(len(options))]
                else:
                    self.policy[:, i] = [options[j]/sum(options) for j in range(len(options))]
            else:
                totalchoices = sum([options[optloop] == max(options)
                                   for optloop in range(len(options))])
                self.policy[:, i] = [(1.0 / totalchoices if options[optloop] == max(options) else 0)
                                     for optloop in range(len(options))]

    def GetStates(self, StartingPoint, ActionSequence):
        # Produce the highest likelihood path given an action sequence.
        # This function produces the true state changes when the transition matrix is binary.
        StateSequence = [0] * (len(ActionSequence) + 1)
        StateSequence[0] = StartingPoint
        for i in range(len(ActionSequence)):
            StateSequence[i + 1] = (
                self.T[StateSequence[i], ActionSequence[i], :]).argmax()
        return StateSequence

    def Run(self, State, Simple=False):
        # Sample an action given the state and return the resulting state.
        # If there is more than one action that is equally valuable
        # the function will select a random one, except when Simple=True
        if Simple:
            ActSample=0
        else:
            ActSample=random.uniform(0,1) 
        ActionProbs = self.policy[:, State]
        ActionChoice = -1
        for j in range(len(ActionProbs)):
            if ActSample < ActionProbs[j]:
                ActionChoice = j
                break
            else:
                ActSample -= ActionProbs[j]
        # Get next state
        EndStates = self.T[State, ActionChoice, :]
        StateSample = random.uniform(0, 1)
        for j in range(len(EndStates)):
            if StateSample < EndStates[j]:
                EndState = j
                break
            else:
                StateSample -= EndStates[j]
        return [EndState, ActionChoice]

    def Display(self, Full=False):
        # Print class properties
        if Full:
            for (property, value) in vars(self).iteritems():
                print property, ': ', value
        else:
            for (property, value) in vars(self).iteritems():
                print property

import MDP
import numpy as np
import math


class Planner(object):

    """
    Planner class handles the MDP and has additional functions to use the MDPs policy.
    Planner contains an Markov Decision Process object (MDP), a softmax parameter (tau), a future discount parameter (gamma), a value iteration convergence parameter (epsilon)
    and the probability that agents may die (deathprob)
    """

    def __init__(self, diagonal=True, MDP=MDP.MDP(), distmat=None, tau=0.001, gamma=0.9999, epsilon=0.00001):
        """
        Build a Planner.

        All arguments are option. If no arguments are provided the object contains an empty MDP with prespecified parameters.

        Args:
        diagonal[boolean] determines if agent can travel diagonally
        MDP         MDP object
        distmat [matrix] Matrix where distmat[i,j] is the minimum cost for moving from state i to state j.
                    This matrix is not exhaustive on the state-space. Instead, it selects the starting and end state as well as states with rewards.
        tau         softmax parameter. Set by default to 0.001
        gamma       future discount parameter. Set by default to 0.9999
        epsilon     value iteration convergence limit. Set by default to 0.00001
        """
        self.MDP = MDP
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.diagonal = diagonal
        self.distmat = distmat

    def BuildCostMatrix(self, Agent, Map):
        """
        Build a matrix with minimal cost between goals.

        Args:
            Agent (agent object)
            Map (map object)
        """
        TargetNo = len(self.Map.Locations) + 2
        DistMat = np.zeros((TargetNo, TargetNo))
        TargetSpace = self.Map.Locations
        TargetSpace.append(self.Map.ExitState)
        for i in range(1, TargetNo):
            TempPolicy = self.PlanForGoal(i)

    def Integrate(self, Agent, Map, ExitType="OneLocation", exitstate=0):
        """
        Integrate builds the MDP using a Map and an Agent object.
        This function calls Planner methods that help build the MDP.
        It doesn't return anything. Instead. the object's MDP will change.
        ARGUMENTS:
        Agent     Agent object
        Map       Map or Map object
        """
        found = 0
        if ExitType == "OneLocation":
            found = 1
            # Agent can only exit from the map in exitstate
            self.MDP = MDP.MDP(range(self.GetDeepStateSize(Map)), Map.A, self.BuildDeepT(
                Agent, Map, [exitstate]), self.BuildDeepR(Agent, Map), self.diagonal, self.gamma)
        if ExitType == "Border":
            found = 1
            # Exit from any map border:
            self.MDP = MDP.MDP(range(self.GetDeepStateSize(Map)), Map.A, self.BuildDeepT(
                Agent, Map), self.BuildDeepR(Agent, Map), self.diagonal, self.gamma)
        if ExitType == "North":
            found = 1
            # Exit from any top location:
            self.MDP = MDP.MDP(range(self.GetDeepStateSize(Map)), Map.A, self.BuildDeepT(
                Agent, Map, False, [range(Map.x)]), self.BuildDeepR(Agent, Map), self.diagonal, self.gamma)
        if found == 0:
            print "ERROR: ExitType does not exit."

    def ComputePolicy(self, Softmax=True):
        """
        ComputePolicy runs value iteration on it's MDP and then builds an optimal policy.
        ARGUMENTS:
        Softmax.   If set to True (Default) the policy will be softmaxed (Using the Planner.tau parameter).
        """
        self.MDP.ValueIteration(self.epsilon)  # Do value iteration on MDP.
        self.MDP.BuildPolicy(self.tau, Softmax)  # Get softmax probabilities.

    def EvalRoute(self, StartingPoint, ActionSequence):
        """
        EvalRoute(StartingPoint,ActionSequence)
        returns the probability of the agent taking the ActionSequence from StartingPoint.
        If there are agents who need help and may die this function is optimistic and assumes no one dies (thus never sending the agent to deathstates)
        """
        # This function retrieves the optimistic state sequence where no
        # objects disappear.
        StateSequence = self.MDP.GetStates(StartingPoint, ActionSequence)
        p = 1  # probability taking the action sequence
        for i in range(len(ActionSequence)):
            p *= (self.MDP.policy[ActionSequence[i], StateSequence[i]])
        return p

    def SimulatePathFor(self, StartingPoint, steps):
        """
        SimulatePathFor(StartingPoint,steps)
        simulates an agent's actions for the specified number of steps starting from the starting point
        """
        Actions = [-1] * steps
        State = StartingPoint
        for i in range(steps):
            [State, Actions[i]] = self.MDP.Run(State)
        return Actions

    def SimulatePathUntil(self, StartingPoint, StopStates, Limit, Simple=False):
        """
        Simulate path from StartingPoint until agent reaches a state in the StopStates list.
        Simulation ends after the agent has taken more steps than specified on Limit.
        """
        iterations = 0
        Actions = []
        StateSequence = [StartingPoint]
        State = StartingPoint
        while State not in StopStates:
            [State, NewAct] = self.MDP.Run(State, Simple)
            Actions.append(NewAct)
            StateSequence.append(State)
            iterations += 1
            if (iterations > Limit):
                return [Actions, StateSequence]
        return [Actions, StateSequence]

    def BuildR(self, Agent, Map):
        """
        Build the cost function for an MDP using the Map and Agent objects.

        Args:
            Agent (Agent object)
            Map (Map object)

        Returns:
            R (matrix): Reward function as a matrix where R[A,S] is the reward for tkaing action A in state S
        """
        R = np.zeros((len(Map.A), len(Map.S)))

        Costs = [-Agent.costs[Map.StateTypes[i]] for i in range(len(Map.S))]

        # Add regular costs to first four actions
        for i in range(4):
            R[i, :] = Costs

        if len(Map.A) > 4:
            Costs = [i * np.sqrt(2) for i in Costs]
            for i in range(4, 8):
                R[i, :] = Costs
        return R

    def UpdateSoftMax(self, tau):
        """
        Update softmax value.

        Args:
        tau (float): Softmax value. Must be greater than 0
        """
        if tau < 0:
            print "ERROR: Cannot use this value to softmax"
            return None
        self.tau = tau

    def UpdateConvergence(self, epsilon):
        """
        Update MDPs convergence value.

        Args:
        epsilon (float): Convergence value for MDP's value iteration
        """
        if epsilon >= 1 or epsilon < 0:
            print "ERROR: Cannot use this value as future discount"
            return None
        self.epsilon = epsilon

    def UpdateDiscount(self, gamma):
        """
        Update MDPs future discount.

        Args:
        gamma (float): Future value. Must be between 0 and 1
        """
        if gamma >= 1 or gamma < 0:
            print "ERROR: Cannot use this value as future discount"
            return None
        self.gamma = gamma

    def Display(self, Full=False):
        """
        Print object attributes.

        .. Internal function::

           This function is for internal use only.

        Args:
            Full (bool): When set to False, function only prints attribute names. Otherwise, it also prints its values.

        Returns:
            standard output summary
        """
        # Print class properties
        if Full:
            for (property, value) in vars(self).iteritems():
                print property, ': ', value
        else:
            for (property, value) in vars(self).iteritems():
                print property

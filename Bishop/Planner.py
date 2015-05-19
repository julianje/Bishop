# -*- coding: utf-8 -*-

"""
Planner class handles the MDP and has additional functions to use the MDPs policy.
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import MDP
import numpy as np
import copy
import sys


class Planner(object):

    def __init__(self, CostMatrix=None, tau=10):
        """
        Build a Planner.

        All arguments are option. If no arguments are provided the object contains an empty MDP with prespecified parameters.

        Args:
        MDPs         main MDP object to use
        policy      list of policies for navigating between goals.
        CostMatrix [matrix] Matrix where CostMatrix[i,j] is the minimum cost for moving from state i to state j.
                    This matrix is not exhaustive on the state-space. Instead, it selects the starting and
                    end state as well as states with rewards.
        tau         softmax parameter. Set by default to 0.001
        gamma       future discount parameter. Set by default to 0.9999
        epsilon     value iteration convergence limit. Set by default to 0.00001
        """
        self.MDP = []
        # Policies is a list of lists, where policies[i][j] contains the MDPs policy from moving from i to j
        # Note that here i and j are not MDP raw state numbers, but the
        # Planners criticalstates (starting point, exit, and states with
        # objects).
        self.Policies = []
        self.CriticalStates = []
        self.tau = tau
        self.CostMatrix = CostMatrix
        # CODE CONSTANTS
        # Internal reward value to plan between goals
        self.planningreward = 500
        self.gamma = 0.95  # Internal future discount to plan between goals

    def BuildPlanner(self, Map, Agent, Validate=True):
        """
        Build the planner using a map, an agent, and an option path.

        Args:
            Map (Map): A map object.
            Agent (Agent): an agent object.
            Validate (bool): Check if objects work
        """
        if Validate:
            # Check that map has all it needs
            if not Map.Validate():
                print "ERROR: Map failed to validate. PLANNER-001"
                return None
            # Check that agent's cost and reward dimensions match map.
            if Agent.CostDimensions != len(np.unique(Map.StateTypes)):
                print "ERROR: Agent's cost dimensions do not match map object. PLANNER-002"
                return None
            if Agent.RewardDimensions != len(set(Map.ObjectLocations)):
                print "ERROR: Agent's reward dimensions do not match map object. PLANNER-003"
                return None
        # Create main MDP object.
        # This assumes that the Map object has a dead exit state.
        # Map's Validate checks this.
        self.MDP = MDP.MDP(
            Map.S + [max(Map.S) + 1], Map.A, Map.T, self.BuildCostFunction(Agent, Map), self.gamma, self.tau)
        self.CriticalStates = [Map.StartingPoint]
        self.CriticalStates.extend(Map.ObjectLocations)
        self.CriticalStates.extend([Map.ExitState])
        # build the costmatrix and store the policies
        [Policies, CostMatrix] = self.Plan(Agent, Map)
        self.Policies = Policies
        self.CostMatrix = CostMatrix

    def Plan(self, Agent, Map, Validate=True):
        """
        Build a plan's cost matrix and store the policies.

        Args:
            Agent (Agent): Agent object
            Map (Map): Map object
            Validate (bool): Check if modifications result in legal MDP objects (Set to True when testing new models)

        Returns
        [Policies, CostMatrix].   Policies stores how to move from states to states and cost matrix stores the cost incurred.
            Policies is a list Policies[i] contains a softmaxed optimal policy (as a numpy array) to move to CriticalStates[j].
            In each policy Pol[i][j] contains the probability of selecting action i in state j
            CostMatrix is a numpy array where CostMatrix[i][j] contains the minimum cost to move from CriticalStates[i] to CriticalStates[j]
        """
        CostMatrix = np.zeros(
            (len(self.CriticalStates), len(self.CriticalStates)))
        Policies = []
        # Now iterate over each combination of critical states.
        # Loop can skip over starting state because agent will never go there.
        print range(1, len(self.CriticalStates))
        for TargetStateIndex in range(1, len(self.CriticalStates)):
            # Duplicate MDP that you'll manipulate
            subMDP = copy.deepcopy(self.MDP)
            # Get target state
            TargetState = self.CriticalStates[TargetStateIndex]
            sys.stdout.write("Targetstate: " + str(TargetState) + "\n")
            # Reroute target state to dead state
            # Set to zeros.
            subMDP.T[TargetState, :, :] = 0
            # Any action sends to dead state
            subMDP.T[TargetState, :, len(Map.S)] = 1
            # Replace with big reward
            subMDP.R[:, TargetState] = self.planningreward
            if Validate:
                subMDP.Validate()
            # Calculate and save optimal policy
            subMDP.ValueIteration()
            subMDP.BuildPolicy()
            Policies.append(copy.deepcopy(subMDP.policy))
            # Loop over all other critical states and use them as starting
            # points
            PotentialStartingPointIndices = range(
                TargetStateIndex) + range(TargetStateIndex + 1, len(self.CriticalStates) - 1)  # The last minus 1 is because we don't need to consider the exit state as a starting point
            sys.stdout.write(
                "\tPotential startingpoint Indices: " + str(PotentialStartingPointIndices) + "\n")
            for OriginalPointIndex in PotentialStartingPointIndices:
                # Get sequence of actions and states
                [Actions, StateSequence] = self.SimulatePathUntil(self.CriticalStates[
                    OriginalPointIndex], self.CriticalStates[TargetStateIndex], subMDP)
                print Actions
                print StateSequence
                # Get the cost associated with each combination of actions and states
                # and sum them to get the total cost.
                TotalCost = sum(
                    [self.MDP.R[Actions[i]][StateSequence[i]] for i in range(len(Actions))])
                CostMatrix[OriginalPointIndex][TargetStateIndex] = TotalCost
                CostMatrix[TargetStateIndex][OriginalPointIndex] = TotalCost
        return [Policies, CostMatrix]

    def SimulatePathUntil(self, StartingPoint, StopStates, inputMDP, Limit=300, Softmax=False, Simple=False):
        """
        Simulate path from StartingPoint until agent reaches a state in the StopStates list.
        Simulation ends after the agent has taken more steps than specified on Limit.

        Args:
            StartingPoint (int): State where agent beings
            StopStates (int or list): State of list of states where simulationg should end
            inputMDP (MDP): MDP object to use
            Softmax (bool): Use softmax when simulating?
            Simple (bool): When set to true the MDP selects the first highest-value action
                            (rather than sampling a random one where more than one action are equally good).
                            Note that simple parameter only makes sense when softmax is off.

        """
        iterations = 0
        Actions = []
        if not isinstance(StopStates, list):
            StopStates = [StopStates]
        StateSequence = [StartingPoint]
        State = StartingPoint
        while State not in StopStates:
            [State, NewAct] = inputMDP.Run(State, Softmax, Simple)
            Actions.append(NewAct)
            StateSequence.append(State)
            iterations += 1
            if (iterations > Limit):
                return [Actions, StateSequence]
        return [Actions, StateSequence]

    def BuildCostFunction(self, Agent, Map, DeadState=True):
        """
        Build the cost function for an MDP using the Map and Agent objects.

        Args:
            Agent (Agent object)
            Map (Map object)
            DeadState (bool): Indicates if it should add a dead state with cost 0

        Returns:
            C (matrix): Cost function as a matrix where C[A,S] is the cost for tkaing action A in state S
        """
        # if not isinstance(Agent, Agent):
        #    print "ERROR: Did not receive correct agent object. PLANNER-004"
        # if not isinstance(Map, Map):
        #    print "ERROR: Did not receive correct map object. PLANNER-005"

        Costs = [-Agent.costs[Map.StateTypes[i]] for i in range(len(Map.S))]

        if DeadState:
            C = np.zeros((len(Map.A), len(Map.S) + 1))
            Costs.append(0)
        else:
            C = np.zeros((len(Map.A), len(Map.S)))
        # Add regular costs to first four actions.
        for i in range(4):
            C[i, :] = Costs
        # If agent can travel diagonally then add the diagonal costs.
        if len(Map.A) > 4:
            Costs = [i * np.sqrt(2) for i in Costs]
            for i in range(4, 8):
                C[i, :] = Costs
        return C

    def EvalRoute(self, StartingPoint, ActionSequence):
        """
        EvalRoute(StartingPoint,ActionSequence)
        returns the probability of the agent taking the ActionSequence from StartingPoint.
        """
        # This function retrieves the optimistic state sequence where no
        # objects disappear.
        StateSequence = self.MDP.GetStates(StartingPoint, ActionSequence)
        p = 1  # probability taking the action sequence
        for i in range(len(ActionSequence)):
            p *= (self.MDP.policy[ActionSequence[i], StateSequence[i]])
        return p

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

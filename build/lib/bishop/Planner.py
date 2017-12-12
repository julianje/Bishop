# -*- coding: utf-8 -*-

"""
Planner class handles the MDP and has additional functions to use the MDPs policy.
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import MDP
import numpy as np
import copy
import math
import random
import sys
import scipy.misc
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
from itertools import product, repeat, permutations


class Planner(object):

    def __init__(self, Agent, Map, Method="Linear", Validate=True):
        """
        Build a Planner.

        All arguments are option. If no arguments are provided the object contains an empty MDP with prespecified parameters.

        Args:
            Agent (Agent): Agent object
            Map (Map): Map object
            Method (str): "Rate" or "Linear" for type of utility function
            Validate (bool): Run object validation? Helps find bugs
        """
        self.Method = Method
        self.Agent = Agent
        self.Map = Map
        # Check if you need to set Agent's capacity
        if self.Agent.Capacity == -1:
            self.Agent.Capacity = len(self.Map.ObjectLocations)
        self.MDP = []
        # Policies is a list of lists, where policies[i] contains the MDPs policy for reaching position CriticalStates[i]
        # Note that here i and j are not MDP raw state numbers, but the
        # Planners criticalstates (starting point, exit, and states with
        # objects).
        self.Policies = []
        self.CriticalStates = []
        self.CostMatrix = []
        # CODE CONSTANTS
        # Internal reward value to plan between goals
        self.planningreward = 500
        self.gamma = 0.95  # Internal future discount to plan between goals
        self.Prepare(Validate)

    def Prepare(self, Validate=True):
        """
        Run the planner and build the utility function.
        This function just helps make other code blocks easier to reader.

        Args:
            Validate (bool): Run validation?
        """
        try:
            self.BuildPlanner(Validate)
            self.ComputeUtilities()
        except Exception as error:
            print(error)

    def BuildPlanner(self, Validate=True):
        """
        Build the planner using the object's map, agent, and path details.

        Args:
            Validate (bool): Check that the objects have all the information to run.
        """
        if Validate:
            # Check that map has all it needs
            if not self.Map.Validate():
                print("ERROR: Map failed to validate. PLANNER-001")
                return None
            # Check that agent's cost and reward dimensions match map.
            if self.Agent.CostDimensions != len(np.unique(self.Map.StateTypes)):
                print(
                    "ERROR: Agent's cost dimensions do not match map object. PLANNER-002")
                return None
            if self.Agent.RewardDimensions != len(set(self.Map.ObjectLocations)):
                print(
                    "ERROR: Agent's reward dimensions do not match map object. PLANNER-003")
                return None
        # Create main MDP object.
        # This assumes that the Map object has a dead exit state.
        # Map's Validate checks this.
        self.MDP = MDP.MDP(
            self.Map.S + [max(self.Map.S) + 1], self.Map.A, self.Map.T, self.BuildCostFunction(), self.gamma, self.Agent.actionTau)
        self.CriticalStates = [self.Map.StartingPoint]
        self.CriticalStates.extend(self.Map.ObjectLocations)
        self.CriticalStates.extend([self.Map.ExitState])
        # build the costmatrix and store the policies
        [Policies, CostMatrix, DistanceMatrix] = self.Plan(Validate)
        self.Policies = Policies
        self.CostMatrix = CostMatrix
        self.DistanceMatrix = DistanceMatrix
        self.Utilities = None
        self.goalindices = None

    def Plan(self, Validate=True):
        """
        Plan how to move between goals, store the policies, estimate the costs, and build the cost matrix.

        .. Warning::

           This function is for internal use only.

        Args:
            Validate (bool): Check if modifications result in legal MDP objects (Set to True when testing new models)

        Returns
        [Policies, CostMatrix, DistanceMatrix].   Policies stores how to move from states to states and cost matrix stores the cost incurred.
            Policies is a list Policies[i] contains a softmaxed optimal policy (as a numpy array) to move to CriticalStates[i].
            In each policy Pol[i][j] contains the probability of selecting action i in state j
            CostMatrix is a numpy array where CostMatrix[i][j] contains the minimum cost to move from CriticalStates[i] to CriticalStates[j]
            DistanceMatrix contains the numerical distance in moving from one point to another.
        """
        CostMatrix = np.zeros(
            (len(self.CriticalStates), len(self.CriticalStates)))
        DistanceMatrix = np.zeros(
            (len(self.CriticalStates), len(self.CriticalStates)))
        # First policy is the one for moving towards starting point. Leave it
        # empty
        Policies = [[]]
        # Now iterate over each combination of critical states.
        # Loop can skip over starting state because agent will never go there.
        for TargetStateIndex in range(1, len(self.CriticalStates)):
            # Duplicate MDP that you'll manipulate
            subMDP = copy.deepcopy(self.MDP)
            # Get target state
            TargetState = self.CriticalStates[TargetStateIndex]
            # Reroute target state to dead state
            # Set to zeros.
            subMDP.T[TargetState, :, :] = 0
            # Any action sends to dead state
            subMDP.T[TargetState, :, len(self.Map.S)] = 1
            # Add a big reward
            subMDP.R[:, TargetState] += self.planningreward
            if Validate:
                subMDP.Validate()
            # Calculate and save optimal policy
            subMDP.ValueIteration()
            subMDP.BuildPolicy(self.Agent.SoftmaxAction)
            Policies.append(copy.deepcopy(subMDP.policy))
            # Loop over all other critical states and use them as starting
            # points
            PotentialStartingPointIndices = range(
                TargetStateIndex) + range(TargetStateIndex + 1, len(self.CriticalStates) - 1)  # The last minus 1 is because we don't need to consider the exit state as a starting point
            for OriginalPointIndex in PotentialStartingPointIndices:
                # Get sequence of actions and states
                [Actions, StateSequence] = self.SimulatePathUntil(self.CriticalStates[
                    OriginalPointIndex], self.CriticalStates[TargetStateIndex], subMDP)
                # Get the cost associated with each combination of actions and states
                # and sum them to get the total cost.
                # Note that the terminology changes a bit here. The utility
                # function is saved inside the MDP's reward function.
                TotalCost = sum(
                    [self.MDP.R[Actions[i]][StateSequence[i]] for i in range(len(Actions))])
                CostMatrix[OriginalPointIndex][TargetStateIndex] = TotalCost
                DistanceMatrix[OriginalPointIndex][TargetStateIndex] = sum([1 if i < 4 else np.sqrt(2) for i in Actions])
        return [Policies, CostMatrix, DistanceMatrix]

    def SimulatePathUntil(self, StartingPoint, StopStates, inputMDP, Limit=300, Simple=False):
        """
        .. Warning::

           This function is for internal use only.

        Simulate path from StartingPoint until agent reaches a state in the StopStates list.
        Simulation ends after the agent has taken more steps than specified on Limit.

        IMPORTANT: THIS FUNCTION USES LOCAL MDPS AND SUPPORT BUILDING THE UTILITY FUNCTION.
        TO SIMULATE THROUGH THE NAIVE UTILITY CALCULUS USE Planner.Simulate()

        Args:
            StartingPoint (int): State where agent beings
            StopStates (int or list): State of list of states where simulationg should end
            inputMDP (MDP): MDP object to use
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
            [State, NewAct] = inputMDP.Run(
                State, self.Agent.SoftmaxAction, Simple)
            Actions.append(NewAct)
            StateSequence.append(State)
            iterations += 1
            if (iterations > Limit):
                print("ERROR: Simulation exceeded timelimit. PLANNER-009")
                return [Actions, StateSequence]
        return [Actions, StateSequence]

    def BuildCostFunction(self, DeadState=True):
        """
        Build the cost function for an MDP using the Map and Agent objects.
        Dead state adds a 0 cost movement towards an absorbing state with reward 0 (which makes valueiteration for the subMDPs converge faster because the agent can only take the reward once).

        Args:
            DeadState (bool): Indicates if it should add a dead state with cost 0.

        Returns:
            C (matrix): Cost function as a matrix where C[A,S] is the cost for tkaing action A in state S.
        """

        Costs = [-self.Agent.costs[self.Map.StateTypes[i]]
                 for i in range(len(self.Map.S))]

        if DeadState:
            C = np.zeros((len(self.Map.A), len(self.Map.S) + 1))
            Costs.append(0)
        else:
            C = np.zeros((len(self.Map.A), len(self.Map.S)))
        # Add regular costs to first four actions.
        for i in range(4):
            C[i, :] = Costs
        # If agent can travel diagonally then add the diagonal costs.
        if len(self.Map.A) > 4:
            Costs = [i * np.sqrt(2) for i in Costs]
            for i in range(4, 8):
                C[i, :] = Costs
        return C

    def ComputeUtilities(self):
        """
        Build the goal space and compute the utility function.

        .. Warning::

           This function is for internal use only.
        """
        # Generate all possible plans
        Targets = self.Map.ObjectLocations
        # Get all possible combinations of the objects
        noneset = zip(range(len(Targets)), repeat(None))
        res = list(product(*noneset))
        subsets = [[x for x in list(i) if x is not None] for i in res]
        # Now add permutations
        goalindices = []
        for i in range(len(subsets)):
            for j in permutations(subsets[i]):
                goalindices.append(list(j))
        # Reduce goal indices to the ones that the agent has capacity for.
        goalindices_temp = [i for i in goalindices if len(
            i) <= self.Agent.Capacity and len(i) >= self.Agent.Minimum]
        # Remove duplicates
        goalindices = []
        for i in goalindices_temp:
            if i not in goalindices:
                goalindices.append(i)
        utility = [0] * len(goalindices)
        # For each sequence of goals
        for i in range(len(goalindices)):
            # add start and exit state
            # Adding 1 to goal indices to adjust for shift in CostMatrix
            # (because it includes the starting state)
            goals = [0] + [j + 1 for j in goalindices[i]] + \
                [len(self.CriticalStates) - 1]
            # Compute the costs
            costs = sum([self.CostMatrix[goals[j - 1], goals[j]]
                         for j in range(1, len(goals))])
            if self.Method == "Discount":
                # Get all the rewards, and then discount the organic ones.
                rewards = [self.Agent.rewards[self.Map.ObjectTypes[j]] for j in goalindices[i]]
                # Extract which rewards are organic.
                organic = [self.Map.Organic[j] for j in goalindices[i]]
                # Now loop over rewards and modify them.
                for currentreward in range(len(organic)):
                    # If it's non-organic. Leave the reward as is.
                    if organic:
                        # Discount the reward
                        # Get the overall distance. Here you need to sum over the whole past path.
                        DistanceTraveled = sum([self.DistanceMatrix[goals[j - 1], goals[j]] for j in range(1, currentreward)])
                        OverallSurvivalProb = self.Map.SurvivalProb ** DistanceTraveled
                        rewards[currentreward] = rewards[currentreward] * OverallSurvivalProb
                        # If the agent dies, the reward becomes a cost. Add the expected cost of the plan:
                        costs += rewards[currentreward] * (1 - OverallSurvivalProb)
                rewards = sum(rewards)
            else:
                # Add the rewards without discounting.
                rewards = sum(
                    [self.Agent.rewards[self.Map.ObjectTypes[j]] for j in goalindices[i]])
            if self.Method == "Rate":
                utility[i] = rewards * 1.0 / abs(costs)  # Make rate positive
            else:
                utility[i] = rewards + costs  # Costs are already negative
        self.Utilities = utility
        self.goalindices = goalindices

    def GetPlanDistribution(self):
        """
        Return the probability distribution of the selected plan. see also GetActionDistribution()
        """
        if self.Utilities is None:
            print("ERROR: Missing utilities. PLANNER-013")
            return None
        if self.goalindices is None:
            print("ERROR: Mising goal space. PLANNER-014")
            return None
        options = self.Utilities
        options = options - abs(max(options))
        try:
            if self.Agent.choiceTau is not None:
                options = [math.exp(options[j] / self.Agent.choiceTau)
                           for j in range(len(options))]
            else:
                options = [1 if options[j] == max(
                    options) else 0 for j in range(len(options))]
        except OverflowError:
            print("ERROR: Failed to softmax utility function. PLANNER-008")
            return None
        if sum(options) == 0:
            # Make uniform distribution
            planprobabilities = [1.0 / len(options)] * len(options)
        else:
            planprobabilities = [options[j] /
                                 sum(options) for j in range(len(options))]
        return(planprobabilities)

    def GetActionDistribution(self):
        """
        Return the probability distribution of the next action by integrating over
        all possible plans.
        That is,

        p(a) = \sum_{plans} p(a|plan)p(plan).

        see also GetPlanDistribution()
        """
        if self.Utilities is None:
            print("ERROR: Missing utilities. PLANNER-013")
            return None
        if self.goalindices is None:
            print("ERROR: Mising goal space. PLANNER-014")
            return None
        options = self.Utilities
        options = options - abs(max(options))
        try:
            if self.Agent.choiceTau is not None:
                options = [math.exp(options[j] / self.Agent.choiceTau)
                           for j in range(len(options))]
            else:
                options = [1 if options[j] == max(
                    options) else 0 for j in range(len(options))]
        except OverflowError:
            print("ERROR: Failed to softmax utility function. PLANNER-008")
            return None
        if sum(options) == 0:
            # Make uniform distribution
            planprobabilities = [1.0 / len(options)] * len(options)
        else:
            planprobabilities = [options[j] /
                                 sum(options) for j in range(len(options))]
        # Now iterate over each plan, get the probability distribution of actions,
        # and weight by the planprobabilities.
        ActionDistribution = [0] * len(self.goalindices)
        CurrentState = [self.CriticalStates[0]]
        for CurrentPlanIndex in range(len(self.goalindices)):
            planindices = [
                0] + [j + 1 for j in self.goalindices[CurrentPlanIndex]] + [len(self.CriticalStates) - 1]
            # [0] is just the starting state, so agent starts by executing first policy after that.
            # Load the policy onto the MDP
            self.MDP.policy = self.Policies[planindices[1]]
            # Get the action distribution vector
            ActionDistribution[CurrentPlanIndex] = self.MDP.policy[
                :, CurrentState]
            # Weight them by the utility.
            ActionDistribution[CurrentPlanIndex] = [ActionProb * planprobabilities[
                CurrentPlanIndex] for ActionProb in ActionDistribution[CurrentPlanIndex]]
        # Now integrate all of them!
        NumberofActions = np.shape(ActionDistribution)[1]
        NumberofPlans = np.shape(ActionDistribution)[0]
        FinalActionDistribution = [0] * NumberofActions
        for i in range(NumberofActions):
            FinalActionDistribution[i] = sum(
                [ActionDistribution[j][i] for j in range(NumberofPlans)])
        return(FinalActionDistribution)

    def Simulate(self, Simple=True):
        """
        Simulate an agent until it reaches the exit state or time runs out.
        IMPORTANT: THIS FUNCTION SIMULATES THROUGH THE NAIVE UTILITY CALCULUS.
        SimulatePathUntil() USES LOCAL MDPS.

        Args:
            Simple (bool): When more than one action is highest value, take the first one?
        """
        if self.Utilities is None:
            print("ERROR: Missing utilities. PLANNER-006")
            return None
        if self.goalindices is None:
            print("ERROR: Missing goal space. PLANNER-007")
            return None
        if self.Agent.SoftmaxChoice:
            options = self.Utilities
            options = options - abs(max(options))
            try:
                options = [
                    math.exp(options[j] / self.Agent.choiceTau) for j in range(len(options))]
            except OverflowError:
                print("ERROR: Failed to softmax utility function. PLANNER-008")
                return None
            if sum(options) == 0:
                # If all utilities are equal
                if Simple:
                    choiceindex = 0
                else:
                    choiceindex = random.choice(range(len(options)))
            else:
                softutilities = [
                    options[j] / sum(options) for j in range(len(options))]
                ChoiceSample = random.uniform(0, 1)
                choiceindex = -1
                for j in range(len(softutilities)):
                    if ChoiceSample < softutilities[j]:
                        choiceindex = j
                        break
                    else:
                        ChoiceSample -= softutilities[j]
        else:
            choiceindex = self.Utilities.index(max(self.Utilities))
        planindices = [0] + [j + 1 for j in self.goalindices[choiceindex]] + \
            [len(self.CriticalStates) - 1]
        # Simulate each sub-plan
        Actions = []
        States = [self.CriticalStates[0]]
        for i in range(1, len(planindices)):
            # Use this policy on local MDP
            self.MDP.policy = self.Policies[planindices[i]]
            [subA, subS] = self.SimulatePathUntil(
                self.CriticalStates[planindices[i - 1]], self.CriticalStates[planindices[i]], self.MDP)
            Actions.extend(subA)
            States.extend(subS[1:])
        return [Actions, States]

    def Likelihood(self, ActionSequence):
        """
        Calculate the loglikelihood of a sequence of actions

        Args:
            ActionSequence (list): List of observed actions
        """
        LogLikelihood = 0
        # Part 1. Decompose action sequence into sub-goals.
        ###################################################
        # Get list of states
        StateSequence = self.MDP.GetStates(
            self.Map.StartingPoint, ActionSequence)
        self.LastStateSquence = StateSequence
        # Get the index of the critical states the agent visited.
        VisitedindicesFull = [
            self.CriticalStates.index(i) if i in self.CriticalStates else -1 for i in StateSequence]
        VisitedindicesFull = filter(lambda a: a != -1, VisitedindicesFull)
        # If agent crosses same spot more than once then
        # only the first pass matters.
        # EXCEPT when you're crossing the exit state. The exit state only
        # matters if it's the final state.
        Visitedindices = []
        for i in VisitedindicesFull:
            # Check that it's new, but that it's not the exit state.
            if (i not in Visitedindices) and (i != len(self.CriticalStates) - 1):
                Visitedindices.append(i)
        # Now add the exit state if it's there.
        if VisitedindicesFull[-1] == len(self.CriticalStates) - 1:
            Visitedindices.append(VisitedindicesFull[-1])
            Complete = True
        else:
            Complete = False
        # Part 2. Compute likelihoods of each complete sub-sequence.
        ############################################################
        # Now switch back to the indices you'll use to call the policies.
        objectscollected = copy.deepcopy(Visitedindices)
        for i in range(1, len(objectscollected)):
            tempPolicy = self.Policies[objectscollected[i]]
            beginstate = StateSequence.index(
                self.CriticalStates[objectscollected[i - 1]])
            endstate = StateSequence.index(
                self.CriticalStates[objectscollected[i]])
            for j in range(beginstate, endstate):
                prob = tempPolicy[ActionSequence[j]][StateSequence[j]]
                if prob > 0:
                    LogLikelihood += np.log(prob)
                else:
                    # If one of the complete subsequences
                    # has probability zero then you can return value
                    # immediately
                    return (-sys.maxint - 1)
        # Part 3. Compute likelihood of selecting that goal.
        ####################################################
        # Get objects the agent has collected
        objectscollected = objectscollected[1:]  # Remove starting point
        if Complete:
            objectscollected.pop()  # Remove exit state
        # objects collected right now contains the indices for the Cost Matrix.
        # Now we want to compare them to the goal indices so we need to
        # subtract one.
        objectscollected = [i - 1 for i in objectscollected]
        # Check that objects collected lies within the range of the map.
        if Complete:
            if (len(objectscollected) < self.Agent.Minimum) or (len(objectscollected) > self.Agent.Capacity):
                print(
                    "\nERROR: Number of objects agent collected is outside the range specified in the map.")
                return None
        else:
            if (len(objectscollected) > self.Agent.Capacity):
                print(
                    "\nERROR: Number of objects agent collected is outside the range specified in the map.")
                return None
        # Find all action sequences that are consistent with the observations:
        if Complete:
            goalindex = [self.goalindices.index(objectscollected)]
        else:
            # If goal is incomplete then select all plans
            # that are consistent with the observed actions
            goalindex = []
            for i in range(len(self.goalindices)):
                if objectscollected == self.goalindices[i][:len(objectscollected)]:
                    goalindex.append(i)
        # Calculate the probability of selecting each goal
        options = self.Utilities
        options = options - abs(max(options))
        try:
            if self.Agent.SoftmaxChoice:
                options = [math.exp(options[j] / self.Agent.choiceTau)
                           for j in range(len(options))]
            else:
                # if not softmaxed just choose the action with the highest
                # value
                bestactions = np.where(options == max(options))
                options[:] = 0
                options[bestactions] = 1
        except OverflowError:
            print("ERROR: Failed to softmax utility function. PLANNER-011")
        if sum(options) == 0:
            softutilities = [1.0 / len(options)] * len(options)
        else:
            softutilities = [
                options[j] / sum(options) for j in range(len(options))]
        # If path is compelte then there is only one goal that is consistent
        # so you just need to add the likelihood and your'e done!
        if Complete:
            # If the path was complete then you can
            # just add the likelihood of the goal and you're done
            if softutilities[goalindex[0]] > 0:
                LogLikelihood += np.log(softutilities[goalindex[0]])
            else:
                # Set to the closest you can get to log(0)
                LogLikelihood = (-sys.maxint - 1)
            return LogLikelihood
        # All code below will only be executed if the path was incomplete
        #################################################################
        # The code below computes P(A|Gi)*P(Gi|C,R) for each goal Gi
        # that is consistent with the past actions. The action sequence A
        # is the sequence happening after the las object was collected
        #
        # Take the probability you already computed.
        LogLikelihoodTerms = [LogLikelihood] * len(goalindex)
        # Now add the utility of the goal to each term
        for i in range(len(goalindex)):
            if softutilities[goalindex[i]] > 0:
                LogLikelihoodTerms[i] += np.log(softutilities[goalindex[i]])
            else:
                # Set to the closest you can get to log(0)
                LogLikelihoodTerms[i] = (-sys.maxint - 1)
        # Get the starting point when target-uncertainty begins
        NewStartingPoint = self.CriticalStates[Visitedindices[-1]]
        # Get the new states
        NewStates = StateSequence[StateSequence.index(NewStartingPoint):]
        # Get the actions the agent took after uncertainty begins
        NewActions = ActionSequence[StateSequence.index(NewStartingPoint):]
        # Check
        if (len(NewStates)) != (len(NewActions) + 1):
            print("ERROR: New states do not align with new actions. PLANNER-012")
            return None
        # For each goal compute the probability of the actions past the last
        # critical state
        for i in range(len(goalindex)):
            Missinggoals = self.goalindices[i][len(objectscollected):]
            if Missinggoals == []:
                nextgoal = len(self.CriticalStates) - 1
            else:
                # Add 1 because StartingPoints is in CriticalStates, so goal
                # indices are shifted by 1.
                nextgoal = Missinggoals[0] + 1
            tempPolicy = self.Policies[nextgoal]
            # Get actions that haven't been accounted for yet
            # Use tempPolicy
            for j in range(len(NewActions)):
                # Only add stuff if you're not on the smallest value yet
                if LogLikelihoodTerms[i] != (-sys.maxint - 1):
                    NewProb = tempPolicy[NewActions[j]][NewStates[j]]
                    if NewProb > 0:
                        LogLikelihoodTerms[i] += np.log(NewProb)
                    else:
                        LogLikelihoodTerms[i] = -sys.maxint - 1
        LogLikelihood = scipy.misc.logsumexp(LogLikelihoodTerms)
        return LogLikelihood

    def GetPivot(self, state):
        """
        Internal function to transform int id's into x-y position of the top-left part of the grid.
        (e.g., (0,0) for state 0 in position (1,1)).

        Args:
            state (int): State id.
        Returns:
            [x,y]: coordinates
        """
        y = state / self.Map.mapwidth
        x = state - y * self.Map.mapwidth
        return [x, y]

    def DrawMap(self, filename, ActionSequence=[], size=20):
        """
        Save map as an image.

        Args:
            filename (String): Name of file for saved image.
            ActionSequence [list]: list of (numeric) actions the agent took.
            size (int): Size of each grid in pixels.
        """
        terraincolors = ["#F0A3FF", "#0075DC", "#993F00", "#4C005C",
                         "#005C31", "#2BCE48", "#FFCC99", "#808080"]
        objectcolors = ["#FF5005", "#FFFF00", "#FFFF80", "#990000",
                        "#740AFF", "#00998F", "#426600", "#003380", "#C20088", "#94FFB5",
                        "#2BCE48", "#5EF1F2"]
        if not self.Map.Validate():
            print(
                "WARNING: Map is not well formed. May fail to render image. PLANNER-013")
        # Draw map first:
        im = Image.new(
            "RGB", [self.Map.mapwidth * size, self.Map.mapheight * size])
        draw = ImageDraw.Draw(im)
        for i in range(self.Map.mapwidth):
            for j in range(self.Map.mapheight):
                dimensions = [
                    i * size, j * size, (i + 1) * size, (j + 1) * size]
                draw.rectangle(dimensions,
                               terraincolors[self.Map.StateTypes[self.Map.mapwidth * j + i]], "#000000")
        # add objects, starting point, and exit.
        fontpath = os.path.dirname(__file__) + '/Fonts/'
        font = ImageFont.truetype(
            os.path.join(fontpath, "estre.ttf"), int(size * 4.0 / 3.0))
        fontsmall = ImageFont.truetype(
            os.path.join(fontpath, "estre.ttf"), int(size * 4.0 / 6.0))
        [SPX, SPY] = self.GetPivot(self.Map.StartingPoint)
        draw.text([SPX * size + (size / 4), SPY * size], "S", "#000000", font)
        [EX, EY] = self.GetPivot(self.Map.ExitState)
        draw.text([EX * size, EY * size], "E", "#000000", font)
        # add objects
        for obj in range(len(self.Map.ObjectLocations)):
            [OX, OY] = self.GetPivot(self.Map.ObjectLocations[obj])
            draw.ellipse(
                [OX * size, OY * size, (OX + 1) * size, (OY + 1) * size], objectcolors[self.Map.ObjectTypes[obj]], "#000000")
        # add path
        if ActionSequence != []:
            [DX, DY] = [SPX, SPY]
            for actionid in range(len(ActionSequence)):
                action = ActionSequence[actionid]
                if action in [0, 4, 6]:
                    DX -= 1
                if action in [1, 5, 7]:
                    DX += 1
                if action in [2, 4, 5]:
                    DY -= 1
                if action in [3, 6, 7]:
                    DY += 1
                if DX != EX or DY != EY:
                    draw.text([DX * size, DY * size, (DX + 1) * size,
                               (DY + 1) * size], str(actionid), "#000000", fontsmall)
        # Save image
        im.save(filename)

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
                print(property, ': ', value)
        else:
            for (property, value) in vars(self).iteritems():
                print(property)

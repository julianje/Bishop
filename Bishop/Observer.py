import numpy as np
import Planner
import sys
import PosteriorContainer
import time

# There are three main functions
# ComputeLikelihood, SimulateAgent, and InferAgent


class Observer(object):

    def __init__(self, M, A):
        self.M = M  # Map
        self.A = A  # Agent
        self.Plr = Planner.Planner(self.M.diagonal)
        self.Plr.Integrate(A, M)
        self.MapName = None

    def AddMapName(self, MapName):
        self.MapName = MapName

    def ComputeLikelihood(self, StartingState, ActionSequence, Softmax=True):
        self.Plr.Integrate(self.A, self.M)
        self.Plr.ComputePolicy(Softmax)
        return self.Plr.EvalRoute(StartingState, ActionSequence)

    def InferAgent(self, StartingCoordinates, ActionSequence, Samples, Softmax=True, CostRestriction=False):
        # Generate different agents and compute their likelihoods
        # CostRestriction determines if first cost should always be the
        # smallest
        StartingPoint = self.GetStartingPoint(StartingCoordinates)
        SampledCosts = np.zeros((Samples, self.A.CostDimensions))
        SampledRewards = np.zeros((Samples, self.A.RewardDimensions))
        SampleLikelihoods = np.zeros((Samples))
        APursuit = np.zeros((Samples))
        BPursuit = np.zeros((Samples))
        for i in range(Samples):
            if i == 0:
                start = time.time()
            # First terrain is not necessarily the easiest.
            self.A.ResampleAgent(CostRestriction)
            # Save all sampled costs, rewards, and likelihoods
            SampledCosts[i] = self.A.costs
            SampledRewards[i] = self.A.rewards
            SampleLikelihoods[i] = (self.ComputeLikelihood(
                StartingPoint, ActionSequence, Softmax))
            StateSequence = self.Plr.MDP.GetStates(
                StartingPoint, ActionSequence)
            # The death state has no information about what the agent did
            # If the agent is already in death state then go back
            # two steps
            if StateSequence[-1] == self.GetExitState():
                CurrentPoint = StateSequence[-2]
            else:
                CurrentPoint = StateSequence[-1]
            [APursuit[i], BPursuit[i]] = self.ComputeChoices(CurrentPoint)
            if i == 0:
                end = time.time()
                secs = (end - start) * Samples
                sys.stdout.write("Expected time: ")
                if (secs < 60):
                    sys.stdout.write(str(round(secs, 2)) + " seconds.\n")
                else:
                    mins = secs * 1.0 / 60
                    if (mins < 60):
                        sys.stdout.write(str(round(mins, 2)) + " minutes.\n")
                    else:
                        hours = mins * 1.0 / 60
                        sys.stdout.write(str(round(hours, 2)) + " hours.\n")
                sys.stdout.flush()
        Normalizer = sum(SampleLikelihoods)
        if Normalizer == 0:
            print "Error: Failed to infer agent. Consider running more samples, lowering softmax, or using a more efficient path."
            return None
        SampleLikelihoods /= sum(SampleLikelihoods)
        Res = PosteriorContainer.PosteriorContainer(
            SampledCosts, SampledRewards, SampleLikelihoods, APursuit, BPursuit)
        # Add meta data to the PosteriorContainer
        Res.AddExtraInfo(StartingCoordinates, self.M.PullTargetStates(
        ), ActionSequence, self.M.GetActionNames(ActionSequence), Softmax)
        Res.AddCostNames(self.M.StateNames)
        Res.AssociateMap(self.MapName)
        return Res

    def ComputeChoices(self, StartingPoint):
        # For each set of costs and reward check if a rational agent
        # would pick up object A and object B
        [Ac, St] = self.SimulateAgent(StartingPoint, False, True)
        APursuit = 0
        BPursuit = 0
        worldsize = self.M.GetWorldSize()
        if (St[len(St) - 2] == worldsize) or (St[len(St) - 2] == worldsize * 3):
            # A was picked up
            APursuit = 1
        if (St[len(St) - 2] == worldsize * 2) or (St[len(St) - 2] == worldsize * 3):
            # B was picked up
            BPursuit = 1
        return([APursuit, BPursuit])

    def SimulateAgent(self, StartingState, Softmax=True, Simple=False):
        # Simple parameter gets sent to the MDP. When set to true the MDP
        # always returns the first highest-value action. Otherwise it
        # returns a sample from the set of highest-value actions.
        self.Plr.Integrate(self.A, self.M)
        self.Plr.ComputePolicy(Softmax)
        [Actions, States] = self.Plr.SimulatePathUntil(
            StartingState, self.GetExitState(), self.M.GetWorldSize() * 2, Simple)
        return [Actions, States]

    def GetExitState(self):
        return [self.Plr.GetDeepStateSize(self.M) - 1]

    def GetStateSequence(self, StartingState, Actions, Numerical=False):
        # Function makes best guess on where the actions will lead.
        print "Warning: This function makes the best guesses. If all noise is in softmax and the transition matrix is deterministic then you're fine."
        print "Also note that Action vector is numeric (instead of having action numbers, set the extra flag to false"
        if Numerical:
            ActionNumbers = Actions
        else:
            ActionNumbers = self.M.GetActionList(Actions)
        States = [StartingState]
        CurrState = StartingState
        for CurrAction in ActionNumbers:
            CurrState = self.Plr.MDP.T[
                CurrState, CurrAction, :].argmax()  # Best guess
            States.append(CurrState % self.M.GetWorldSize())
        return States

    def SimulateAgents(self, StartingCoordinates, Samples, HumanReadable=False, Simple=True, Softmax=False, ConstrainTerrains=False):
        """
        Simulate agents navigating through the map.

        Attributes:
        StartingCoordinates [x,y]    Agent's starting point
        HumanReadable [boolean] Outputs actions in their numeric values or their names
        Samples [int]          Number of agents to generate
        Softmax [boolean]      Marks wether to softmax actions. False by default
        Simple [boolean]       Sometimes the agent can take more than one action.
                               When this happens the agent will randomly select one of the actions.
                               When simple is set to true the agent will take the first action on the set. True by default. 
        ConstrainTerrains      [boolean] When set to true the samples force the first terrain to be less costly than the rest. False by default.
        """
        if Softmax:
            if Simple == True:
                print "WARNING: Can't do simple sampling when softmax is on. Turning Simple off"
                Simple = False
        sys.stdout.write("ObjectA,ObjectB,")
        discard = [sys.stdout.write(str(i) + ",") for i in self.M.StateNames]
        sys.stdout.write("Actions\n")
        StartingPoint = self.GetStartingPoint(StartingCoordinates)
        for i in range(Samples):
            self.A.ResampleAgent(ConstrainTerrains)
            [Actions, States] = self.SimulateAgent(
                StartingPoint, Softmax, Simple)
            for j in range(self.A.RewardDimensions):
                sys.stdout.write(str(self.A.rewards[j]) + ",")
            for j in range(self.A.CostDimensions):
                sys.stdout.write(str(self.A.costs[j]) + ",")
            if (States[-1] == self.Plr.GetDeepStateSize(self.M) - 1):
                if HumanReadable:
                    print self.M.GetActionNames(Actions)
                else:
                    print Actions
            else:
                sys.stdout.write("AGENT FAILED TO REACH EXIT")

    def GetStartingPoint(self, StartingCoordinates):
        return self.M.GetRawStateNumber(StartingCoordinates)

    def GetActionList(self, Actions):
        return self.M.GetActionList(Actions)

    def GetSemantics(self):
        # Get names of states and actions
        print self.M.ActionNames
        print self.M.LocationNames
        print self.M.StateNames

    def Display(self, Full=False):
        # Print class properties
        if Full:
            for (property, value) in vars(self).iteritems():
                print property, ': ', value
        else:
            for (property, value) in vars(self).iteritems():
                print property

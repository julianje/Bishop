import numpy as np
import Planner
import sys

# There are three main functions
# ComputeLikelihood, SimulateAgent, and InferAgent

class Observer(object):

    def __init__(self, M, A):
        self.M = M #Map
        self.A = A #Agent
        self.Plr = Planner.Planner()
        self.Plr.Integrate(A, M)

    def ComputeLikelihood(self, StartingState, ActionSequence, Softmax=True):
        self.Plr.Integrate(self.A, self.M)
        self.Plr.ComputePolicy(Softmax)
        return self.Plr.EvalRoute(StartingState, ActionSequence)

    def InferAgent(self, StartingState, ActionSequence, Samples, Softmax=True, CostRestriction=False):
        # Generate different agents and compute their likelihoods
        # CostRestriction determines if first cost should always be the smallest
        SampledCosts = np.zeros((Samples, self.A.CostDimensions))
        SampledRewards = np.zeros((Samples, self.A.RewardDimensions))
        SampleLikelihoods = np.zeros((Samples))
        APursuit = np.zeros((Samples))
        BPursuit = np.zeros((Samples))
        for i in range(Samples):
            self.A.ResampleAgent(CostRestriction) # First terrain is not necessarily the easiest.
            # Save all sampled costs, rewards, and likelihoods
            SampledCosts[i] = self.A.costs
            SampledRewards[i] = self.A.rewards
            SampleLikelihoods[i] = (self.ComputeLikelihood(
                StartingState, ActionSequence, Softmax))
            [APursuit[i],BPursuit[i]]=self.ComputeChoices(StartingState)
        SampleLikelihoods /= sum(SampleLikelihoods)
        return [SampledCosts, SampledRewards, SampleLikelihoods, APursuit, BPursuit]

    def ComputeChoices(self,StartingPoint):
        # For each set of costs and reward check if a rational agent
        # would pick up object A and object B
        [Ac,St]=self.SimulateAgent(StartingPoint,False,True)
        APursuit=0
        BPursuit=0
        worldsize=self.M.GetWorldSize()
        if (St[len(St)-2]==worldsize) or (St[len(St)-2]==worldsize*3):
            # A was picked up
            APursuit=1
        if (St[len(St)-2]==worldsize*2) or (St[len(St)-2]==worldsize*3):
            # B was picked up
            BPursuit=1
        return([APursuit,BPursuit])

    def SimulateAgent(self, StartingState, Softmax=True):
        self.Plr.Integrate(self.A, self.M)
        self.Plr.ComputePolicy(Softmax)
        [Actions, States] = self.Plr.SimulatePathUntil(StartingState, self.GetExitState(), self.M.GetWorldSize()*2)
        return [Actions, States]

    def GetExitState(self):
        return [self.Plr.GetDeepStateSize(self.M)-1]

    def GetStateSequence(self, StartingState, Actions, Numerical=False):
        # Function makes best guess on where the actions will lead.
        print "Warning: This function makes the best guesses. If all noise is in softmax and the transition matrix is deterministic then you're fine."
        print "Also note that Action vector is numeric (instead of having action numbers, set the extra flag to false"
        if (type(self.M) == Map.Map):
            # Call Map function if it's a simple map.
            return self.M.GetStateSequence(StartingState, Actions)
        else:
            # Otherwise...
            if Numerical:
                ActionNumbers = Actions
            else:
                ActionNumbers = self.M.GetActionList(Actions)
            States = [StartingState]
            CurrState = StartingState
            for CurrAction in ActionNumbers:
                CurrState = self.Plr.MDP.T[CurrState, CurrAction, :].argmax() # Best guess
                States.append(CurrState % self.M.GetWorldSize())
            return States

    def SimulateAgents(self,StartingPoint,Samples,Softmax=False,ContrainTerrains=False):
        for i in range(Samples):
            self.A.ResampleAgent(ContrainTerrains)
            [Actions, States] = self.SimulateAgent(StartingPoint,Softmax)
            for j in range(self.A.RewardDimensions):
                sys.stdout.write(str(self.A.rewards[j])+",")
            for j in range(self.A.CostDimensions):
                sys.stdout.write(str(self.A.costs[j])+",")
            if (States[-1]==self.Plr.GetDeepStateSize(self.M)-1):
                print Actions
            else:
                sys.stdout.write("AGENT FAILED TO REACH EXIT")

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

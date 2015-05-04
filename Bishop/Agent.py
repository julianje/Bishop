import random
import numpy as np


class Agent(object):

    """Agent class.
    Create an agent who has costs and rewards.

    Attributes:
    CostDimensions [int]   Dimension of cost simplex
    RewardDimensions [int] Dimensions of reward simplex
    costs [list]           list with sampled costs
    rewards [list]         list with sampled rewards
    """

    def __init__(self, Map, CostParam=0.1, RewardParam=10):
        # ARGUMENTS:
        #
        # Map          MAP object
        # CostParam    Parameter used for sampling costs.
        # RewardParam  Parameter used for sampling rewards.
        # Both parameters change meaning depending on the distribution you use.

        self.CostDimensions = len(np.unique(Map.StateTypes))
        # Get dimensions over which you'll build your simplex
        self.RewardDimensions = sum([i > 0 for i in map(len, Map.Locations)])
        if self.RewardDimensions == 0:
            print "WARNING: No rewards on map."
        self.CostParam = CostParam
        self.RewardParam = RewardParam
        self.ResampleCosts()  # Generate random cost of map
        self.ResampleRewards()  # Generate random rewards for objects

    def ResampleAgent(self, Apathy=0, Restrict=False):
        # Generate a random agent by changing it's competence and motivation
        # If Restrict is set to true the agent assumes that the first terrain
        # is always the least costly
        self.ResampleCosts(Apathy)
        self.ResampleRewards(Apathy)
        if Restrict:
            temp = self.costs[0]
            new = self.costs.argmin()
            minval = self.costs[new]
            self.costs[new] = temp
            self.costs[0] = minval

    def ResampleCosts(self, Apathy=0):
        # Resample the agent's competence
        self.costs = self.Sample(
            self.CostDimensions, self.CostParam, Kind="Exponential")
        self.costs = [0 if random.random() < Apathy else i for i in self.costs]

    def ResampleRewards(self, Apathy=0):
        # Resample the agent's preferences
        self.rewards = self.Sample(
            self.RewardDimensions, self.RewardParam, Kind="Exponential")
        self.rewards = [
            0 if random.random() < Apathy else i for i in self.rewards]

    def Sample(self, dimensions, SamplingParam, Kind):
        # Generate a random sample from different distributions.
        # ARGUMENTS: Dimensions and sample kind
        if(Kind == "Simplex"):
            # Output: Simplex sample of length 'dimensions' (Adds to 1)
            sample = -np.log(np.random.rand(dimensions))
            return sample / sum(sample)
        if(Kind == "Exponential"):
            sample = [
                np.random.exponential(SamplingParam) for j in range(dimensions)]
            return sample

    def Display(self, Full=True):
        # Print class properties
        if Full:
            for (property, value) in vars(self).iteritems():
                print property, ': ', value
        else:
            for (property, value) in vars(self).iteritems():
                print property

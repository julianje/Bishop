# -*- coding: utf-8 -*-

"""
Markov Decision Process solver.
"""

__license__ = "MIT"

import random
import numpy as np


class Agent(object):

    def __init__(self, Map, Prior="Exponential", CostParam=0.1, RewardParam=10):
        """
        Agent class.

        Create an agent who has costs and rewards.

        Args:
            Map (Map): A map object.
            Prior (str): String indicating prior's name (Only Simplex and Exponential supported)
            CostParam (float): Parameter for sampling costs
            RewardParam (float): Parameter for sampling rewards
        """
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
        """
        Reset the agent with random parameters.

        Args:
            Apathy (float): Probability that agent doesn't like objects.
            Restrict (bool): If true, first terrain is always the least costly

        Returns:
            None
        """
        self.ResampleCosts(Apathy)
        self.ResampleRewards(Apathy)
        if Restrict:
            temp = self.costs[0]
            new = self.costs.argmin()
            minval = self.costs[new]
            self.costs[new] = temp
            self.costs[0] = minval

    def ResampleCosts(self, Apathy=0):
        """
        Reset agent's costs.

        Args:
            Apathy (float): Probability that agent doesn't like objects.

        Returns:
            None
        """
        # Resample the agent's competence
        self.costs = self.Sample(
            self.CostDimensions, self.CostParam, Kind=self.Prior)
        self.costs = [0 if random.random() < Apathy else i for i in self.costs]

    def ResampleRewards(self, Apathy=0):
        """
        Reset agent's rewards.

        Args:
            Apathy (float): Probability that agent doesn't like objects.

        Returns:
            None
        """
        # Resample the agent's preferences
        self.rewards = self.Sample(
            self.RewardDimensions, self.RewardParam, Kind=self.Prior)
        self.rewards = [
            0 if random.random() < Apathy else i for i in self.rewards]

    def Sample(self, dimensions, SamplingParam, Kind):
        """
        Generate a sample from some distribution

        Args:
            dimensions (int): Number of dimensions
            SamplingParam: Parameter to use on distribution
            Kind (str): Name of distribution

        Returns:
            None
        """
        if(Kind == "Simplex"):
            # Output: Simplex sample of length 'dimensions' (Adds to 1)
            sample = -np.log(np.random.rand(dimensions))
            return sample / sum(sample)
        if(Kind == "Exponential"):
            sample = [
                np.random.exponential(SamplingParam) for j in range(dimensions)]
            return sample

    def Display(self, Full=True):
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

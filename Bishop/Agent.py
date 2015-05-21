# -*- coding: utf-8 -*-

"""
Stores information about agent and comes with supporting methods to sample random agents.
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import random
import numpy as np


class Agent(object):

    def __init__(self, Map, Prior, CostParams, RewardParams, SoftmaxChoice=True, SoftmaxAction=True, choiceTau=8, actionTau=0.01):
        """
        Agent class.

        Create an agent with a set of costs and rewards.
        If sampling parameters are numbers rather than lists the constructor fixes this automatically.

        Args:
            Map (Map): A map object.
            Prior (str): String indicating prior's name. Run Agent.Priors() to see list
            CostParams (list): List of parameters for sampling costs.
            RewardParams (list): List of parameters for sampling rewards.
            SoftmaxChoice (bool): Does the agent select goals optimally?
            SoftmaxAction (bool): Does the agent act upong goals optimally?
            choiceTau (float): Softmax parameter for goal selection.
            actionTau (float): Softmax parameter for action planning.
        """
        self.Prior = Prior
        self.CostDimensions = len(np.unique(Map.StateTypes))
        # Get dimensions over which you'll build your simplex
        self.RewardDimensions = len(set(Map.ObjectTypes))
        self.SoftmaxChoice = SoftmaxChoice
        self.SoftmaxAction = SoftmaxAction
        self.actionTau = actionTau
        self.choiceTau = choiceTau
        if self.RewardDimensions == 0:
            print "WARNING: No rewards on map. AGENT-001"
        if isinstance(CostParams, list):
            self.CostParams = CostParams
        else:
            self.CostParams = [CostParams]
        if isinstance(RewardParams, list):
            self.RewardParams = RewardParams
        else:
            self.RewardParams = [RewardParams]
        self.ResampleCosts()  # Generate random cost of map
        self.ResampleRewards()  # Generate random rewards for objects

    def ResampleAgent(self, Apathy=0, Restrict=False):
        """
        Reset agent with random costs and rewards.

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
            self.CostDimensions, self.CostParams, Kind=self.Prior)
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
            self.RewardDimensions, self.RewardParams, Kind=self.Prior)
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
        if (Kind == "Simplex"):
            # Output: Simplex sample of length 'dimensions' (Adds to 1)
            sample = -np.log(np.random.rand(dimensions))
            return sample / sum(sample)
        if (Kind == "ScaledUniform"):
            return np.random.rand(dimensions) * SamplingParam[0]
        if (Kind == "Gaussian"):
            return np.random.normal(SamplingParam[0], SamplingParam[1], dimensions)
        if (Kind == "Exponential"):
            sample = [
                np.random.exponential(SamplingParam[0]) for j in range(dimensions)]
            return sample

    def Priors(self):
        """
        Print list of supported priors. This is hardcoded for now.
        """
        print ("Simplex, ScaledUniform, Gaussian, Exponential")

    def Display(self, Full=True):
        """
        Print object attributes.

        .. Warning::

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

# -*- coding: utf-8 -*-

"""
Stores information about agent and comes with supporting methods to sample random agents.
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import random
import numpy as np


class Agent(object):

    def __init__(self, Map, CostPrior, RewardPrior, CostParams, RewardParams, Capacity=-1, Minimum=0, SoftmaxChoice=True, SoftmaxAction=True, choiceTau=1, actionTau=0.01, CNull=0, RNull=0, Restrict=False):
        """
        Agent class.

        Create an agent with a set of costs and rewards.
        If sampling parameters are numbers rather than lists the constructor fixes this automatically.

        Args:
            Map (Map): A map object.
            CostPrior (str): String indicating prior's name. Run Agent.Priors() to see list
            RewardPrior (str): String indicating Reward prior's name. Run Agent.Priors() to see list
            CostParams (list): List of parameters for sampling costs.
            RewardParams (list): List of parameters for sampling rewards.
            Capacity (int): Number of objects agent can carry. If set to -1 Planner adjusts
                            it to the total number of objects in the map.
            Minimum (int): Minimum number of objects must take before leaving.
            SoftmaxChoice (bool): Does the agent select goals optimally?
            SoftmaxAction (bool): Does the agent act upong goals optimally?
            choiceTau (float): Softmax parameter for goal selection.
            actionTau (float): Softmax parameter for action planning.
            CNull (float): Probability that a terrain has no cost.
            RNull (float): Probability that an object has no reward
            Restrict (bool): When set to true the cost samples make the first terrain
                            always less costly than the rest.
        """
        # Check that priors exist.
        Priors = self.Priors(False)
        if CostPrior in Priors:
            self.CostPrior = CostPrior
        else:
            print("WARNING: Cost prior not found! Setting to uniform")
            self.CostPrior = "ScaledUniform"
        if RewardPrior in Priors:
            self.RewardPrior = RewardPrior
        else:
            print("WARNING; Reward prior not found! Setting to uniform")
            self.RewardPrior = "ScaledUniform"
        self.Restrict = Restrict
        self.CostDimensions = len(np.unique(Map.StateTypes))
        # Get dimensions over which you'll build your simplex
        self.RewardDimensions = len(set(Map.ObjectTypes))
        self.Capacity = Capacity
        self.Minimum = Minimum
        self.SoftmaxChoice = SoftmaxChoice
        self.SoftmaxAction = SoftmaxAction
        if SoftmaxAction:
            self.actionTau = actionTau
        else:
            self.actionTau = None
        if SoftmaxChoice:
            self.choiceTau = choiceTau
        else:
            self.choiceTau = None
        if self.RewardDimensions == 0:
            print("WARNING: No rewards on map. AGENT-001")
        if isinstance(CostParams, list):
            self.CostParams = CostParams
        else:
            self.CostParams = [CostParams]
        if isinstance(RewardParams, list):
            self.RewardParams = RewardParams
        else:
            self.RewardParams = [RewardParams]
        self.CNull = CNull
        self.RNull = RNull
        self.ResampleCosts()  # Generate random cost of map
        self.ResampleRewards()  # Generate random rewards for objects

    def ResampleAgent(self):
        """
        Reset agent with random costs and rewards.

        Args:
            Restrict (bool): If true, first terrain is always the least costly

        Returns:
            None
        """
        self.ResampleCosts()
        self.ResampleRewards()
        if self.Restrict:
            temp = self.costs[0]
            new = self.costs.argmin()
            minval = self.costs[new]
            self.costs[new] = temp
            self.costs[0] = minval

    def ResampleCosts(self):
        """
        Reset agent's costs.
        """
        # Resample the agent's competence
        self.costs = self.Sample(
            self.CostDimensions, self.CostParams, Kind=self.CostPrior)
        self.costs = [
            0 if random.random() <= self.CNull else i for i in self.costs]

    def ResampleRewards(self):
        """
        Reset agent's rewards.

        Returns:
            None
        """
        # Resample the agent's preferences
        self.rewards = self.Sample(
            self.RewardDimensions, self.RewardParams, Kind=self.RewardPrior)
        self.rewards = [
            0 if random.random() <= self.RNull else i for i in self.rewards]

    def Sample(self, dimensions, SamplingParam, Kind):
        """
        Generate a sample from some distribution

        Args:
            dimensions (int): Number of dimensions
            SamplingParam (list): Parameter to use on distribution
            Kind (str): Name of distribution

        Returns:
            None
        """
        if (Kind == "Simplex"):
            # Output: Simplex sample of length 'dimensions' (Adds to 1)
            sample = -np.log(np.random.rand(dimensions))
            return sample / sum(sample)
        if (Kind == "IntegerUniform"):
            return np.round(np.random.rand(dimensions) * SamplingParam[0])
        if (Kind == "ScaledUniform"):
            return np.random.rand(dimensions) * SamplingParam[0]
        if (Kind == "Gaussian"):
            return np.random.normal(SamplingParam[0], SamplingParam[1], dimensions)
        if (Kind == "Exponential"):
            return [np.random.exponential(SamplingParam[0]) for j in range(dimensions)]
        if (Kind == "Constant"):
            return [0.5 * SamplingParam[0]] * dimensions
        if (Kind == "Beta"):
            return [np.random.beta(SamplingParam[0], SamplingParam[1]) for i in range(dimensions)]
        if (Kind == "Empirical"):
            return [random.choice(SamplingParam) for i in range(dimensions)]
        if (Kind == "PartialUniform"):
            # Generate random samples and scale them by the first parameter.
            samples = np.random.rand(dimensions) * SamplingParam[0]
            # Now iterate over the sampling parameters and push in static
            # values.
            for i in range(1, len(SamplingParam)):
                if SamplingParam[i] != -1:
                    samples[i - 1] = SamplingParam[i]
            return samples
        if (Kind == "PartialGaussian"):
            # Generate random gaussian samples.
            samples = np.random.normal(
                SamplingParam[0], SamplingParam[1], dimensions)
            # Now iterate over the sampling parameters and push in static
            # values.
            for i in range(2, len(SamplingParam)):
                if SamplingParam[i] != -1:
                    samples[i - 2] = SamplingParam[i]
            samples = [0 if i < 0 else i for i in samples]
            return samples

    def Priors(self, human=True):
        """
        Print list of supported priors.

        PRIORS:
        Simplex: No arguments needed.
        IntegerUniform: argument is one real/int that scales the vector
        ScaledUniform: argument is one real/int that scales the vector
        Gaussian: First argument is the mean and second argument is the standard deviation.
        PartialGaussian: First two arguments are the same as Gaussian. Next there should be N arguments
            with N = number of terrains. When Nth value after the main to is -1, the terrain cost gets sampled
            from the gaussian distribution, when the value is different it is held constant. See also PartialUniform.
        Exponential: First parameter is lambda
        Constant: First parameter is the constant value.
        Beta: First parameter is alpha and second is beta.
        PartialUniform: First parameter as a real/int that scales the vector. This argument should be followed by a list of numbers
            that matches the number of terrains. If the entry for terrain i is -1 that terrain get resampled and scaled, if it contains any value
            then the terrain is left constant at that value. E.g., a two terrain world with a PartialUniform prior and parameters 10 -1 0.25
            generates priors where the first terrain is uniform between 0 and 10, and the second paramter is always 0.25

        Args:
            human (bool): If true function prints names, otherwise it returns a list.
        """
        Priors = ['Simplex', 'IntegerUniform', 'ScaledUniform', 'Beta',
                  'Gaussian', 'PartialGaussian', 'Exponential', 'Constant', 'Empirical', 'PartialUniform']
        if human:
            for Prior in Priors:
                print(Prior)
        else:
            return Priors

    def GetSamplingParameters(self):
        """
        Return cost and reward sampling parameters, respectively
        """
        return [self.CostParams, self.RewardParams]

    def SetCostSamplingParams(self, samplingparams):
        """
        Set sampling parameters for costs
        """
        if len(samplingparams) != len(self.CostParams):
            print("Vector of parameters is not the right size")
        else:
            self.CostParams = samplingparams

    def SetRewardSamplingParams(self, samplingparams):
        """
        Set sampling parameters for costs
        """
        if len(samplingparams) != len(self.RewardParams):
            print("Vector of parameters is not the right size")
        else:
            self.RewardParams = samplingparams

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
        if Full:
            for (property, value) in vars(self).iteritems():
                print(property, ': ', value)
        else:
            for (property, value) in vars(self).iteritems():
                print(property)

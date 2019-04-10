# -*- coding: utf-8 -*-

"""
Observer class stores a planner and wraps some higher-level methods around it.
Its main functions are SimulateAgents and InferAgent
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import numpy as np
import Planner
import sys
import math
import PosteriorContainer
import AgentSimulation
import AuxiliaryFunctions
import scipy.misc
from scipy.stats.stats import pearsonr


class Observer(object):

    def __init__(self, A, M, Method="Linear", Validate=False):
        """
        Build an observed object

        Args:
            A (Agent): Agent object
            M (Map): Map objects
            Method (str): What type of planner? "Rate" or "Linear"
            Validate (bool): Should objects be validated?
        """
        self.Plr = Planner.Planner(A, M, Method, Validate)
        self.Validate = Validate
        # hidden variables for progress bar
        self.begincolor = '\033[91m'
        self.endcolor = '\033[0m'
        self.block = u'\u2588'

    def TestModel(self, Simulations, Samples, Return=False, Verbose=True):
        """
        Simulate N agents, infer their parameters, and then correlate the inferred values with the true values.

        Simulations (int): Number of agents to Simulate
        Samples (int): Number of samples to use in each simulation
        Return (bool): When set to true the function returns the data
        Verbose (bool): Print progress bar?
        """
        if Verbose is False and Return is False:
            sys.stdout.write(
                "ERROR: The function is set on silent and return no input.")
            return None
        if Verbose:
            sys.stdout.write("Simulating agents...\n")
            sys.stdout.flush()
        Agents = self.SimulateAgents(Simulations, False, True, False, True)
        if Verbose:
            sys.stdout.write("\n\nRunning inference...\n")
            sys.stdout.flush()
        InferredCosts = [0] * Simulations
        InferredRewards = [0] * Simulations
        for i in range(Simulations):
            if Verbose:
                Percentage = round(i * 100.0 / Simulations, 2)
                sys.stdout.write("\rInferring agent " + str(i + 1) + " |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            Results = self.InferAgent(Agents.Actions[i], Samples)
            InferredCosts[i] = Results.GetExpectedCosts()
            InferredRewards[i] = Results.GetExpectedRewards()
        if Verbose:
            # Print complete progress bar
            sys.stdout.write("\rInferring agent " + str(Simulations) + " |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
            sys.stdout.write("\n")
            # Calculate correlations
            TrueCosts = [item for sublist in Agents.Costs for item in sublist]
            TrueRewards = [
                item for sublist in Agents.Rewards for item in sublist]
            InferenceCosts = [
                item for sublist in InferredCosts for item in sublist]
            InferenceRewards = [
                item for sublist in InferredRewards for item in sublist]
            sys.stdout.write(
                "Costs correlation: " + str(pearsonr(TrueCosts, InferenceCosts)[0]) + "\n")
            sys.stdout.write(
                "Rewards correlation: " + str(pearsonr(TrueRewards, InferenceRewards)[0]) + "\n\n")
        # For each sample get the sequence of actions with the highest
        # likelihood
        if Verbose:
            sys.stdout.write(
                "Using inferred expected values to predict actions...\n")
        PredictedActions = [0] * Simulations
        MatchingActions = [0] * Simulations
        for i in range(Simulations):
            if Verbose:
                Percentage = round(i * 100.0 / Simulations, 2)
                sys.stdout.write("\rSimulating agent " + str(i + 1) + " |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            self.Plr.Agent.costs = InferredCosts[i]
            self.Plr.Agent.rewards = InferredRewards[i]
            self.Plr.Prepare(self.Validate)
            result = self.Plr.Simulate()
            PredictedActions[i] = result[0]
            if result[0] == Agents.Actions[i]:
                MatchingActions[i] = 1
        if Verbose:
            # Print complete progress bar
            sys.stdout.write("\rSimulating agent " + str(Simulations) + " |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
            sys.stdout.write("\n")
            sys.stdout.write(str(sum(MatchingActions) * 100.00 / Simulations) +
                             "% of inferences produced the observed actions.\n")
        if Return:
            InferredAgents = AgentSimulation.AgentSimulation(
                InferredCosts, InferredRewards, PredictedActions, None)
            return [Agents, InferredAgents, MatchingActions]
        else:
            return None

    def DrawMap(self, filename, ActionSequence=[], size=20):
        """
        Save map as an image.

        Args:
            filename (String): Name of file for saved image.
            Actionsequence [list]: list of (numeric) actions the agent took.
            size (int): Size of each grid in pixels.
        """
        if not all(isinstance(x, int) for x in ActionSequence):
            if all(isinstance(x, str) for x in ActionSequence):
                ActionSequence = self.Plr.Map.GetActionList(ActionSequence)
            else:
                print(
                    "ERROR: Action sequence must contains the indices of actions or their names.")
                return None
        self.Plr.DrawMap(filename, ActionSequence, size)

    def GetCR(self):
        """
        Print current costs and rewards
        """
        sys.stdout.write("Costs (" + str(self.Plr.Map.StateNames) +
                         ")" + str(self.Plr.Agent.costs) + "\n")
        sys.stdout.write("Rewards (" + str(self.Plr.Map.ObjectNames) +
                         ")" + str(self.Plr.Agent.rewards) + "\n")

    def SetCR(self, costs, rewards):
        """
        Set costs and rewards.

        Args:
            costs (list): List of cost values
            rewards (list): List of reward values
        """
        if len(costs) == len(self.Plr.Agent.costs):
            self.Plr.Agent.costs = costs
        else:
            print("Cost list does not match number of terrains.")
            return None
        if len(rewards) == len(self.Plr.Agent.rewards):
            self.Plr.Agent.rewards = rewards
        else:
            print("Reward list does not match number of terrains.")
            return None
        self.Plr.Prepare(self.Validate)

    def ComputeProbabilityOfChange(self, ActionSequence, PC, TestVariable, Conditioning, Tolerance=None, Feedback=True):
        """
        This function returns the probability that an agent was knowledgeable or ignorant
        about a cost or a reward, conditioned on them being nowledgeable about one or more sources
        of costs and rewards.

        Args:
            ActionSequence (list): Sequence of actions
            PC (PosteriorContainer): PosteriorContainer object
            TestVariable (string): Random variable to test. Must exist in both containers.
            Conditioning (list of strings): Random variable names to fix across events. Must exist in both containers.
            Tolerance (int): How many decimal points should be left when rounding? When Tolerance=None samples aren't rounded.
            Feedback (bool): Verbose?
        """
        R = self.UpdateExperience(
            ActionSequence, PC, Conditioning, Feedback)
        return AuxiliaryFunctions.ProbabilityOfChange(PC, R[0], TestVariable, Tolerance)

    def UpdateExperience(self, ActionSequence, PC, Conditioning, Normalize=True, Feedback=True):
        """
        This function returns the probability that an agent was knowledgeable or ignorant
        about a cost or a reward, conditioned on them being knowledgeable about one or more sources
        of costs and rewards.

        Args:
            ActionSequence (list): Sequence of actions
            PC (PosteriorContainer): PosteriorContainer
            Conditioning (list of strings): Random variable names to fix across events. Must exist in both containers.
            Normalize (bool): Normalize samples?
            Feedback (bool): Verbose?
        """
        if not all(isinstance(x, int) for x in ActionSequence):
            if all(isinstance(x, str) for x in ActionSequence):
                ActionSequence = self.Plr.Map.GetActionList(ActionSequence)
            else:
                print(
                    "ERROR: Action sequence must contains the indices of actions or their names.")
                return None
        Samples = PC.Samples
        Costs = [0] * Samples
        Rewards = [0] * Samples
        LogLikelihoods = [0] * Samples
        # Find the indices of the dimensions we're locking down (i.e. the agent
        # already knows them).
        RIndices = []
        CIndices = []
        for ConditioningVar in Conditioning:
            if ConditioningVar in PC.ObjectNames:
                RIndices.append(PC.ObjectNames.index(ConditioningVar))
            else:
                CIndices.append(PC.CostNames.index(ConditioningVar))
        if Feedback:
            sys.stdout.write("\n")
        for i in range(Samples):
            if Feedback:
                Percentage = round(i * 100.0 / Samples, 2)
                sys.stdout.write("\rProgress |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            # Resample the agent
            self.Plr.Agent.ResampleAgent()
            # and overwrite sample sections where the agent could not have
            # updated (dimension we're conditioning on).
            self.Plr.Agent.costs = [PC.CostSamples[i, j] if j in CIndices else self.Plr.Agent.costs[
                j] for j in range(len(self.Plr.Agent.costs))]
            self.Plr.Agent.rewards = [PC.RewardSamples[i, j] if j in RIndices else self.Plr.Agent.rewards[
                j] for j in range(len(self.Plr.Agent.rewards))]
            # save samples
            Costs[i] = self.Plr.Agent.costs
            Rewards[i] = self.Plr.Agent.rewards
            # Replan
            self.Plr.Prepare(self.Validate)
            # Get log-likelihood
            LogLik = self.Plr.Likelihood(ActionSequence)
            # If anything went wrong stop
            if LogLik is None:
                print("ERROR: Failed to compute likelihood. OBSERVER-001")
                return None
            LogLikelihoods[i] = LogLik
        # Finish printing progress bar
        if Feedback:
            # Print complete progress bar
            sys.stdout.write("\rProgress |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
        if Normalize:
            # Normalize LogLikelihoods
            NormalizeConst = scipy.misc.logsumexp(LogLikelihoods)
            if np.exp(NormalizeConst) == 0:
                sys.stdout.write("\nWARNING: All likelihoods are 0.\n")
            NormLogLikelihoods = LogLikelihoods - NormalizeConst
        else:
            # Hacky way because otherwise the subtraction is on different
            # object types
            NormalizeConst = scipy.misc.logsumexp([0])
            NormLogLikelihoods = LogLikelihoods - NormalizeConst
        Results = PosteriorContainer.PosteriorContainer(np.matrix(Costs), np.matrix(
            Rewards), NormLogLikelihoods, ActionSequence, self.Plr)
        if Feedback:
            sys.stdout.write("\n\n")
            Results.Summary()
            sys.stdout.write("\n")
        # By this point, PC and Results, contain samples.
        return [Results, Conditioning]

    def SetCostSamplingParams(self, samplingparams):
        """
        Set sampling parameters for costs
        """
        self.Plr.Agent.SetCostSamplingParams(samplingparams)

    def SetRewardSamplingParams(self, samplingparams):
        """
        Set sampling parameters for costs
        """
        self.Plr.Agent.SetRewardSamplingParams(samplingparams)

    def InferAgentUsingPC(self, ActionSequence, PC, Combine=True, Normalize=True, Feedback=False):
        """
        Compute the posterior of an action sequence using a set of samples from a PC and their loglikelihoods.
        This let's you take the posterior from one map and use it as a prior for another map, or simply to
        obtain the likelihoods for two events using the same samples.

        Args:
            ActionSequence (list): Sequence of actions
            PC (PosteriorContainer): PosteriorContainer object
            Combine (bool): When true, the posterior container's loglikelihoods are used as the prior.
                            when false, only the samples are re-used.
            Normalize (bool): Normalize samples?
            Feedback (bool): When true, function gives feedback on percentage complete.
        """
        if not all(isinstance(x, int) for x in ActionSequence):
            if all(isinstance(x, str) for x in ActionSequence):
                ActionSequence = self.Plr.Map.GetActionList(ActionSequence)
            else:
                print(
                    "ERROR: Action sequence must contains the indices of actions or their names.")
                return None
        Samples = PC.Samples
        Costs = [0] * Samples
        Rewards = [0] * Samples
        LogLikelihoods = [0] * Samples
        # Find what samples we already have.
        RIndices = [PC.ObjectNames.index(
            i) if i in PC.ObjectNames else -1 for i in self.Plr.Map.ObjectNames]
        CIndices = [PC.CostNames.index(
            i) if i in PC.CostNames else -1 for i in self.Plr.Map.StateNames]
        if Feedback:
            sys.stdout.write("\n")
        for i in range(Samples):
            if Feedback:
                Percentage = round(i * 100.0 / Samples, 2)
                sys.stdout.write("\rProgress |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            # Resample the agent
            self.Plr.Agent.ResampleAgent()
            # and overwrite sample sections that we already have
            self.Plr.Agent.costs = [PC.CostSamples[i, CIndices[
                j]] if CIndices[j] != -1 else self.Plr.Agent.costs[j] for j in range(len(self.Plr.Agent.costs))]
            self.Plr.Agent.rewards = [PC.RewardSamples[i, RIndices[
                j]] if RIndices[j] != -1 else self.Plr.Agent.rewards[j] for j in range(len(self.Plr.Agent.rewards))]
            # save samples
            Costs[i] = self.Plr.Agent.costs
            Rewards[i] = self.Plr.Agent.rewards
            # Replan
            self.Plr.Prepare(self.Validate)
            # Get log-likelihood
            LogLik = self.Plr.Likelihood(ActionSequence)
            # If anything went wrong just stop
            if LogLik is None:
                print("ERROR: Failed to compute likelihood. OBSERVER-001")
                return None
            # Add the prior
            if Combine:
                prior = PC.LogLikelihoods[i]
                if (LogLik == (-sys.maxint - 1) or prior == (-sys.maxint - 1)):
                    LogLikelihoods[i] = (-sys.maxint - 1)
                else:
                    if ((LogLik + prior) < (-sys.maxint - 1)):
                        LogLikelihoods[i] = (-sys.maxint - 1)
                    else:
                        LogLikelihoods[i] = LogLik + prior
            else:
                LogLikelihoods[i] = LogLik
        # Finish printing progress bar
        if Feedback:
            # Print complete progress bar
            sys.stdout.write("\rProgress |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
        if Normalize:
            # Normalize LogLikelihoods
            NormalizeConst = scipy.misc.logsumexp(LogLikelihoods)
            if np.exp(NormalizeConst) == 0:
                sys.stdout.write("\nWARNING: All likelihoods are 0.\n")
            NormLogLikelihoods = LogLikelihoods - NormalizeConst
        else:
            # Hacky way because otherwise the subtraction is on different
            # object types
            NormalizeConst = scipy.misc.logsumexp([0])
            NormLogLikelihoods = LogLikelihoods - NormalizeConst
        Results = PosteriorContainer.PosteriorContainer(np.matrix(Costs), np.matrix(
            Rewards), NormLogLikelihoods, ActionSequence, self.Plr)
        if Feedback:
            sys.stdout.write("\n\n")
            Results.Summary()
            sys.stdout.write("\n")
        return Results

    def PredictPlan(self, PC, CSV=False, Feedback=False):
        """
        Return a probability distribution of the agent's plan.
        Use PredictionAction() to predict a single action.

        Args:
            PC (PosteriorContainer): PosteriorContainer object.
            Feedback (bool): When true, function gives feedback on percentage complete.
            Samples (int): Number of samples to use.
            CSV (bool): When set to true, function returns output as a csv rather than returning the values
        """
        Samples = PC.Samples
        Costs = [0] * Samples
        Rewards = [0] * Samples
        PredictedPlans = [0] * len(self.Plr.Utilities)
        # Find what samples we already have.
        RIndices = [PC.ObjectNames.index(
            i) if i in PC.ObjectNames else -1 for i in self.Plr.Map.ObjectNames]
        CIndices = [PC.CostNames.index(
            i) if i in PC.CostNames else -1 for i in self.Plr.Map.StateNames]
        if Feedback:
            sys.stdout.write("\n")
        for i in range(Samples):
            if Feedback:
                Percentage = round(i * 100.0 / Samples, 2)
                sys.stdout.write("\rProgress |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            # Resample the agent
            self.Plr.Agent.ResampleAgent()
            # and overwrite sample sections that we already have
            self.Plr.Agent.costs = [PC.CostSamples[i, CIndices[
                j]] if CIndices[j] != -1 else self.Plr.Agent.costs[j] for j in range(len(self.Plr.Agent.costs))]
            self.Plr.Agent.rewards = [PC.RewardSamples[i, RIndices[
                j]] if RIndices[j] != -1 else self.Plr.Agent.rewards[j] for j in range(len(self.Plr.Agent.rewards))]
            # save samples
            Costs[i] = self.Plr.Agent.costs
            Rewards[i] = self.Plr.Agent.rewards
            # Replan
            self.Plr.Prepare(self.Validate)
            # Get predicted actions
            PlanDistribution = self.Plr.GetPlanDistribution()
            # Get the probability
            probability = np.exp(PC.LogLikelihoods[i])
            # Add all up
            PredictedPlans = [PlanDistribution[
                x] * probability + PredictedPlans[x] for x in range(len(PredictedPlans))]
        # Finish printing progress bar
        if Feedback:
            # Print complete progress bar
            sys.stdout.write("\rProgress |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
        # PredictedPlans is a list of arrays. Make it a list of integers.
        #PredictedPlans = [PredictedPlans[i][0] for i in range(len(PredictedPlans))]
        if not CSV:
            return [self.Plr.goalindices, PredictedPlans]
        else:
            stringgoalindices = [str(x) for x in self.Plr.goalindices]
            print ",".join(stringgoalindices)
            probs = [str(i) for i in PredictedPlans]
            print ",".join(probs)

    def PredictAction(self, PC, CSV=False, Feedback=False):
        """
        Return a probability distribution of the agent's next action.
        Use PredictPlan() to predict the overall plan.

        Args:
            PC (PosteriorContainer): PosteriorContainer object.
            Feedback (bool): When true, function gives feedback on percentage complete.
            Samples (int): Number of samples to use.
            CSV (bool): When set to true, function returns output as a csv rather than returning the values
        """
        Samples = PC.Samples
        Costs = [0] * Samples
        Rewards = [0] * Samples
        PredictedActions = [0] * len(self.Plr.MDP.A)
        # Find what samples we already have.
        RIndices = [PC.ObjectNames.index(
            i) if i in PC.ObjectNames else -1 for i in self.Plr.Map.ObjectNames]
        CIndices = [PC.CostNames.index(
            i) if i in PC.CostNames else -1 for i in self.Plr.Map.StateNames]
        if Feedback:
            sys.stdout.write("\n")
        for i in range(Samples):
            if Feedback:
                Percentage = round(i * 100.0 / Samples, 2)
                sys.stdout.write("\rProgress |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            # Resample the agent
            self.Plr.Agent.ResampleAgent()
            # and overwrite sample sections that we already have
            self.Plr.Agent.costs = [PC.CostSamples[i, CIndices[
                j]] if CIndices[j] != -1 else self.Plr.Agent.costs[j] for j in range(len(self.Plr.Agent.costs))]
            self.Plr.Agent.rewards = [PC.RewardSamples[i, RIndices[
                j]] if RIndices[j] != -1 else self.Plr.Agent.rewards[j] for j in range(len(self.Plr.Agent.rewards))]
            # save samples
            Costs[i] = self.Plr.Agent.costs
            Rewards[i] = self.Plr.Agent.rewards
            # Replan
            self.Plr.Prepare(self.Validate)
            # Get predicted actions
            ActionDistribution = self.Plr.GetActionDistribution()
            # Get the probability
            probability = np.exp(PC.LogLikelihoods[i])
            # Add all up
            PredictedActions = [ActionDistribution[
                x] * probability + PredictedActions[x] for x in range(len(PredictedActions))]
        # Finish printing progress bar
        if Feedback:
            # Print complete progress bar
            sys.stdout.write("\rProgress |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
        # PredictedActions is a list of arrays. Make it a list of integers.
        PredictedActions = [PredictedActions[i][0]
                            for i in range(len(PredictedActions))]
        if not CSV:
            return [self.Plr.Map.ActionNames, PredictedActions]
        else:
            print ",".join(self.Plr.Map.ActionNames)
            probs = [str(i) for i in PredictedActions]
            print ",".join(probs)

    def InferAgent(self, ActionSequence, Samples, Feedback=False, Normalize=True):
        """
        Compute a series of samples with their likelihoods.

        Args:
            ActionSequence (list): Sequence of actions
            Samples (int): Number of samples to use
            Feedback (bool): When true, function gives feedback on percentage complete.
            Normalize (bool): Normalize log-likelihoods? When normalized the LogLikelihoods, integrated
                over matching samples give you the posterior.
        """
        ActionSequence = self.GetActionIDs(ActionSequence)
        return self.InferAgent_ImportanceSampling(ActionSequence, Samples, Normalize, Feedback)

    def GetActionIDs(self, ActionSequence):
        if not all(isinstance(x, int) for x in ActionSequence):
            if all(isinstance(x, str) for x in ActionSequence):
                return self.Plr.Map.GetActionList(ActionSequence)
            else:
                print(
                    "ERROR: Action sequence must contains the indices of actions or their names.")
                return None
        return ActionSequence

    def FindHit(self, ActionSequence, Limit):
        """
        Sample costs and rewards and only produce higher likelihoods

        Args:
        ActionSequence (list): Sequence of actions
        Limit (int): Number of samples to search for
        """
        ActionSequence = self.GetActionIDs(ActionSequence)
        ML = 0
        # Print header
        sys.stdout.write("Sample\t")
        if self.Plr.Map.StateNames is not None:
            for i in self.Plr.Map.StateNames:
                sys.stdout.write(str(i) + "\t")
        else:
            for i in range(self.Plr.Agent.CostDimensions):
                sys.stdout.write("Terrain" + str(i) + "\t")
        if self.Plr.Map.ObjectNames is not None:
            for i in self.Plr.Map.ObjectNames:
                sys.stdout.write(str(i) + "\t")
        else:
            for i in range(self.Plr.Map.RewardDimensions):
                sys.stdout.write("Object" + str(i) + "\t")
        sys.stdout.write("Likelihood\n")
        for i in range(Limit):
            self.Plr.Agent.ResampleAgent()
            self.Plr.Prepare(self.Validate)
            Loglik = self.Plr.Likelihood(ActionSequence)
            if np.exp(Loglik) > ML:
                ML = np.exp(Loglik)
                sys.stdout.write(str(i + 1) + "\t")
                for j in range(self.Plr.Agent.CostDimensions):
                    sys.stdout.write(
                        str(np.round(self.Plr.Agent.costs[j], 2)) + "\t")
                for j in range(self.Plr.Agent.RewardDimensions):
                    sys.stdout.write(
                        str(np.round(self.Plr.Agent.rewards[j], 2)) + "\t")
                sys.stdout.write(str(ML) + "\n")

    def InferAgent_ImportanceSampling(self, ActionSequence, Samples, Normalize=True, Feedback=False):
        """
        Compute a series of samples with their likelihoods using importance sampling

        Args:
            ActionSequence (list): Sequence of actions
            Samples (int): Number of samples to use
            Normalize (bool): Normalize LogLikelihoods when done?
            Feedback (bool): When true, function gives feedback on percentage complete.
        """
        if not all(isinstance(x, int) for x in ActionSequence):
            if all(isinstance(x, str) for x in ActionSequence):
                ActionSequence = self.Plr.Map.GetActionList(ActionSequence)
            else:
                print(
                    "ERROR: Action sequence must contains the indices of actions or their names.")
                return None
        Costs = [0] * Samples
        Rewards = [0] * Samples
        LogLikelihoods = [0] * Samples
        if Feedback:
            sys.stdout.write("\n")
        for i in range(Samples):
            if Feedback:
                Percentage = round(i * 100.0 / Samples, 2)
                sys.stdout.write("\rProgress |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            # Propose a new sample
            self.Plr.Agent.ResampleAgent()
            Costs[i] = self.Plr.Agent.costs
            Rewards[i] = self.Plr.Agent.rewards
            # Replan
            self.Plr.Prepare(self.Validate)
            # Get log-likelihood
            LogLikelihoods[i] = self.Plr.Likelihood(ActionSequence)
            # If anything went wrong just stop
            if LogLikelihoods[i] is None:
                print("ERROR: Failed to compute likelihood. OBSERVER-001")
                return None
        # Finish printing progress bar
        if Feedback:
            # Print complete progress bar
            sys.stdout.write("\rProgress |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
        if Normalize:
            # Normalize LogLikelihoods
            NormalizeConst = scipy.misc.logsumexp(LogLikelihoods)
            if np.exp(NormalizeConst) == 0:
                sys.stdout.write("\nWARNING: All likelihoods are 0.\n")
            NormLogLikelihoods = LogLikelihoods - NormalizeConst
        else:
            # Hacky way because otherwise the subtraction is on different
            # object types
            NormalizeConst = scipy.misc.logsumexp([0])
            NormLogLikelihoods = LogLikelihoods - NormalizeConst
        Results = PosteriorContainer.PosteriorContainer(np.matrix(Costs), np.matrix(
            Rewards), NormLogLikelihoods, ActionSequence, self.Plr)
        if Feedback:
            sys.stdout.write("\n\n")
            Results.Summary()
            sys.stdout.write("\n")
        return Results

    def LL(self, ActionSequence, costs=[], rewards=[]):
        """
        Calcualte the log-likelihood of a sequence of actions given a set of costs and rewards.

        Args:
            AcitonSequence (list): List of observed actions
            costs (list): List of the cost of each terrain
            rewards (list): List of rewards for each object
        """
        if not all(isinstance(x, int) for x in ActionSequence):
            if all(isinstance(x, str) for x in ActionSequence):
                ActionSequence = self.Plr.Map.GetActionList(ActionSequence)
            else:
                print(
                    "ERROR: Action sequence must contains the indices of actions or their names.")
                return None
        if costs != []:
            if len(costs) != self.Plr.Agent.CostDimensions:
                print("ERROR: Number of cost samples does not match number of terrains")
                return None
            else:
                self.Plr.Agent.costs = costs
        if rewards != []:
            if len(rewards) != self.Plr.Agent.RewardDimensions:
                print(
                    "ERROR: Number of reward samples does not match number of object types")
                return None
            else:
                self.Plr.Agent.rewards = rewards
        self.Plr.Prepare(self.Validate)
        return self.Plr.Likelihood(ActionSequence)

    def DrawSimulations(self, Samples, IndexSaving=0, Prefix=""):
        """
        Simulate agents and generate images of the resulting simulations.
        Samples determines how many simulations to run.
        When IndexSaving is true the images are saved numerically. This creates a lot of potential duplicate paths.
        When IndexSaving is false, the image names indicate the paths, so there are no duplicate paths.

        Prefix is a string that gets prefixed onto all saved images.
        """
        sys.stdout.write("Running simulation...\n")
        Res = self.SimulateAgents(Samples, 0, 1, 0, 1)
        # Draw results
        for i in range(len(Res.Actions)):
            if IndexSaving:
                self.DrawMap(Prefix + str(i) + ".png", Res.Actions[i])
            else:
                self.DrawMap(
                    Prefix + str(Res.Actions[i]) + ".png", Res.Actions[i])

    def SimulateAgents(self, Samples, HumanReadable=False, ResampleAgent=True, Simple=True, Verbose=True, replan=True):
        """
        Simulate agents navigating through the map.

        Args:
            Samples (int): Number of agents to Simulate
            HumanReadable (bool): When true, function prints action names rather than action ids.
            ResampleAgent (bool): When false simulation uses the same agent.
                                  If agent is not softmaxing utilities or actions then all action sequences will be identical.
            Simple (bool): When the agent finds more than one equally-valuable action is takes one at random.
                            If simple is set to true it instead chooses the first action in the set.
                            This avoid generating a lot of equivalent paths that look superficially different.

        Returns:
            AgentSimulation object with stored samples, actions, and state transitions.
        """
        Costs = [0] * Samples
        Rewards = [0] * Samples
        Actions = [0] * Samples
        States = [0] * Samples
        for i in range(Samples):
            if Verbose:
                Percentage = round(i * 100.0 / Samples, 2)
                sys.stdout.write("\rProgress |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(
                    self.begincolor + self.block * roundper + self.endcolor)
                sys.stdout.write(" " * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            # Replan
            if replan:
                self.Plr.Prepare(self.Validate)
            # Simulate
            [A, S] = self.Plr.Simulate(Simple)
            # Store results
            Costs[i] = self.Plr.Agent.costs
            Rewards[i] = self.Plr.Agent.rewards
            if HumanReadable:
                Actions[i] = self.Plr.Map.GetActionNames(A)
            else:
                Actions[i] = A
            States[i] = S
            # Reset agent
            if ResampleAgent:
                self.Plr.Agent.ResampleAgent()
        # Finish printing progress bar
        if Verbose:
            # Print complete progress bar
            sys.stdout.write("\rProgress |")
            sys.stdout.write(self.begincolor + self.block * 20 + self.endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.write("\n")
            sys.stdout.flush()
        return AgentSimulation.AgentSimulation(Costs, Rewards, Actions, States, self.Plr.Map.ObjectNames, self.Plr.Map.StateNames)

    def SetStartingPoint(self, StartingPoint, Verbose=True):
        """
        Shorter syntax to change the Map's starting point.

        Args:
            StartingPoint (int): Id of map state with new starting point.
            Verbose (bool): If True, function prints revised map.
        """
        self.Plr.Map.AddStartingPoint(StartingPoint)
        if Verbose:
            self.PrintMap()

    def GetSemantics(self, Complete=False):
        """
        Print action, object, and terrain names.

        Args:
            Complete (bool): If true, function calls Map's print map function instead
        """
        if Complete:
            self.Plr.Map.PrintMap()
        else:
            sys.stdout.write(
                "Action names: " + str(self.Plr.Map.ActionNames) + "\n")
            sys.stdout.write(
                "Object names: " + str(self.Plr.Map.ObjectNames) + "\n")
            sys.stdout.write(
                "Terrain Names: " + str(self.Plr.Map.StateNames) + "\n")

    def PrintMap(self, terrain='*'):
        """
        Shortcut to call the Map object's PrintMap function.

        Args:
            terrain (Character): Character to mark terrains.

        >> Observer.PrintMap()
        >> Observer.PrintMap("X") # Mark empty terrains with 'X'
        """
        self.Plr.Map.PrintMap(terrain)

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
        if Full:
            for (property, value) in vars(self).iteritems():
                print(property, ': ', value)
        else:
            for (property, value) in vars(self).iteritems():
                print(property)

# -*- coding: utf-8 -*-

"""
Observer class stores a planner and wraps some higher-level methods on it.
It's main functions are SimulateAgents and InferAgent
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import numpy as np
import Planner
import sys
import math
import PosteriorContainer
import AgentSimulation
import scipy.misc
from scipy.stats.stats import pearsonr


class Observer(object):

    def __init__(self, A, M, Validate=False):
        """
        Build an observed object

        Args:
            A (Agent): Agent object
            M (Map): Map objects
            Validate (bool): Should objects be validated?
        """
        self.Plr = Planner.Planner(A, M, Validate)
        self.Validate = Validate

    def TestModel(self, Simulations, Samples, Verbose=True):
        """
        Simulate N agents, infer their parameters, and then correlate the inferred values with the true values.

        Simulations (int): Number of agents to Simulate
        Samples (int): Number of samples to use in each simulation
        Verbose (bool): Print progress bar?
        """
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
                begincolor = '\033[91m'
                endcolor = '\033[0m'
                block = u'\u2588'
                space = " "
                Percentage = round(i * 100.0 / Simulations, 2)
                sys.stdout.write("\rInferring agent " + str(i + 1) + " |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(begincolor + block * roundper + endcolor)
                sys.stdout.write(space * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            Results = self.InferAgent(Agents.Actions[i], Samples)
            InferredCosts[i] = Results.GetExpectedCosts()
            InferredRewards[i] = Results.GetExpectedRewards()
        if Verbose:
            # Print complete progress bar
            sys.stdout.write("\rInferring agent " + str(Simulations) + " |")
            sys.stdout.write(begincolor + block * 20 + endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
        sys.stdout.write("\n")
        # Calculate correlations
        TrueCosts = [item for sublist in Agents.Costs for item in sublist]
        TrueRewards = [item for sublist in Agents.Rewards for item in sublist]
        InferenceCosts = [
            item for sublist in InferredCosts for item in sublist]
        InferenceRewards = [
            item for sublist in InferredRewards for item in sublist]
        sys.stdout.write(
            "Costs correlation: " + str(pearsonr(TrueCosts, InferenceCosts)[0]) + "\n")
        sys.stdout.write(
            "Rewards correlation: " + str(pearsonr(TrueRewards, InferenceRewards)[0]) + "\n")

    def InferAgent(self, ActionSequence, Samples, Feedback=False):
        """
        Compute a series of samples with their likelihoods.

        Args:
            ActionSequence (list): Sequence of actions
            Samples (int): Number of samples to use
            Feedback (bool): When true, function gives feedback on percentage complete.
        """
        if not all(isinstance(x, int) for x in ActionSequence):
            if all(isinstance(x, str) for x in ActionSequence):
                ActionSequence = self.Plr.Map.GetActionList(ActionSequence)
            else:
                print "ERROR: Action sequence must contains the indices of actions or their names."
                return None
        Costs = [0] * Samples
        Rewards = [0] * Samples
        LogLikelihoods = [0] * Samples
        if Feedback:
            sys.stdout.write("\n")
        for i in range(Samples):
            if Feedback:
                begincolor = '\033[91m'
                endcolor = '\033[0m'
                block = u'\u2588'
                space = " "
                Percentage = round(i * 100.0 / Samples, 2)
                sys.stdout.write("\rProgress |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(begincolor + block * roundper + endcolor)
                sys.stdout.write(space * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            # Propose a new sample
            self.Plr.Agent.ResampleAgent()
            Costs[i] = self.Plr.Agent.costs
            Rewards[i] = self.Plr.Agent.rewards
            # Replan
            self.Plr.BuildPlanner(self.Validate)
            self.Plr.ComputeUtilities()
            # Get log-likelihood
            LogLikelihoods[i] = self.Plr.Likelihood(ActionSequence)
            # If anything went wrong just stope
            if LogLikelihoods[i] is None:
                print "ERROR: Failed to compute likelihood. OBSERVER-001"
                return None
        # Finish printing progress bar
        if Feedback:
            # Print complete progress bar
            sys.stdout.write("\rProgress " + str(Samples) + " |")
            sys.stdout.write(begincolor + block * 20 + endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.flush()
        # Normalize LogLikelihoods
        NormLogLikelihoods = LogLikelihoods - \
            scipy.misc.logsumexp(LogLikelihoods)
        Results = PosteriorContainer.PosteriorContainer(np.matrix(Costs), np.matrix(
            Rewards), NormLogLikelihoods, ActionSequence, self.Plr)
        if Feedback:
            sys.stdout.write("\n\n")
            Results.Summary()
            sys.stdout.write("\n")
        return Results

    def SimulateAgents(self, Samples, HumanReadable=False, ResampleAgent=True, Simple=True, Verbose=True):
        """
        Simulate agents navigating through the map.

        Args:
            Samples (int): Number of agents to Simulate
            HumanReadable (bool): When true, function prints action names rather than action ids.
            ResampleAgent (bool): When true simulation uses the same agent.
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
                begincolor = '\033[91m'
                endcolor = '\033[0m'
                block = u'\u2588'
                space = " "
                Percentage = round(i * 100.0 / Samples, 2)
                sys.stdout.write("\rProgress |")
                roundper = int(math.floor(Percentage / 5))
                sys.stdout.write(begincolor + block * roundper + endcolor)
                sys.stdout.write(space * (20 - roundper))
                sys.stdout.write("| " + str(Percentage) + "%")
                sys.stdout.flush()
            # Replan
            self.Plr.BuildPlanner(self.Validate)
            self.Plr.ComputeUtilities()
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
            sys.stdout.write(begincolor + block * 20 + endcolor)
            sys.stdout.write("| 100.0%")
            sys.stdout.write("\n")
            sys.stdout.flush()
        return AgentSimulation.AgentSimulation(Costs, Rewards, Actions, States)

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
                print property, ': ', value
        else:
            for (property, value) in vars(self).iteritems():
                print property

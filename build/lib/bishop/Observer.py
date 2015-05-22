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
import PosteriorContainer
import scipy.misc


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

    def InferAgent(self, ActionSequence, Samples, Feedback=False):
        """
        Compute a series of samples with their likelihoods.

        Args:
            ActionSequence (list): Sequence of actions
            Samples (int): Number of samples to use
            Feedback (bool): When true, function gives feedback on percentage complete.
        """
        Costs = [0] * Samples
        Rewards = [0] * Samples
        LogLikelihoods = [0] * Samples
        for i in range(Samples):
            if Feedback:
                sys.stdout.write("\r"+str(round(i*100.0/Samples, 2)) + "% complete")
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
        # Normalize LogLikelihoods
        NormLogLikelihoods = LogLikelihoods - scipy.misc.logsumexp(LogLikelihoods)
        if Feedback:
            sys.stdout.write("\n")
        return PosteriorContainer.PosteriorContainer(np.matrix(Costs), np.matrix(Rewards), NormLogLikelihoods, ActionSequence, self.Plr)

    def SimulateAgents(self, Samples, HumanReadable=False, ResampleAgent=True, Simple=True):
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
        """
        sys.stdout.write("Costs,Rewards,Actions,States\n")
        for i in range(Samples):
            # Reset agent
            if ResampleAgent:
                self.Plr.Agent.ResampleAgent()
            # Replan
            self.Plr.BuildPlanner(False)  # use True to run validation on each object
            self.Plr.ComputeUtilities()
            # Simulate
            [A, S] = self.Plr.Simulate(Simple)
            sys.stdout.write(str(self.Plr.Agent.costs)+","+str(self.Plr.Agent.rewards)+",")
            if HumanReadable:
                sys.stdout.write(str(self.Plr.Map.GetActionNames(A))+",")
            else:
                sys.stdout.write(str(A)+",")
            sys.stdout.write(str(S)+"\n")

    def GetSemantics(self, Complete=False):
        """
        Print action, object, and terrain names.

        Args:
            Complete (bool): If true, function calls Map's print map function instead
        """
        if Complete:
            self.M.PrintMap()
        else:
            sys.stdout.write("Action names: "+str(self.M.ActionNames)+"\n")
            sys.stdout.write("Object names: "+str(self.M.ObjectNames)+"\n")
            sys.stdout.write("Terrain Names: "+str(self.M.StateNames)+"\n")

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

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
import time


class Observer(object):

    def __init__(self, A, M):
        self.Plr = Planner.Planner(A, M)

    def InferAgent(self, ActionSequence, Samples):
        return None

    def SimulateAgents(self, Samples, HumanReadable=False, Simple=True):
        """
        Simulate agents navigating through the map.

        Args:
            Samples (int): Number of agents to Simulate
            HumanReadable (bool): When true, function prints action names rather than action ids.
            Simple (bool): When the agent finds more than one equally-valuable action is takes one at random.
                            If simple is set to true it instead chooses the first action in the set.
                            This avoid generating a lot of equivalent paths that look superficially different.
        """
        sys.stdout.write("Costs,Rewards,Actions,States\n")
        for i in range(Samples):
            # Reset agent
            self.Plr.Agent.ResampleAgent()
            # Replan
            self.Plr.BuildPlanner(False)  # Switch to true to force validation on each object
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

# -*- coding: utf-8 -*-

"""
Store results from agent simulations
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import sys


class AgentSimulation(object):

    def __init__(self, Costs, Rewards, Actions, States):
        """
        Create an object that stores simulation results.

        Args:
            Costs (list): List of cost samples
            Rewards (list): List of reward samples
            Actions (list): List of action sequences
            States (list): List of state transitions
        """
        self.Costs = Costs
        self.Rewards = Rewards
        self.Actions = Actions
        self.States = States

    def PrintActions(self):
        """
        Pretty print action simulations
        """
        for i in range(len(self.Actions)):
            sys.stdout.write(str(self.Actions[i])+"\n")

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
                print property, ': ', value
        else:
            for (property, value) in vars(self).iteritems():
                print property

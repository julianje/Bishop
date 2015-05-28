# -*- coding: utf-8 -*-

"""
Store results from agent simulations
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"


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

# -*- coding: utf-8 -*-

"""
Markov Decision Process solver.
"""

__license__ = "MIT"

import numpy as np
import math
import random


class MDP(object):

    def __init__(self, S=[], A=[], T=[], R=[], gamma=0.1, tau=0.01):
        """
        Markov Decision Process (MDP) class.

        Args:
            S (list): List of states
            A (list): List of actions
            T (matrix): Transition matrix where T[SO,A,SF] is the probability of moving from So to SF after taking action A
            R (list): Reward function
            gamma (float): Future discount
            tau (float): Softmax parameter

        Returns:
            MDP object
        """
        self.S = S
        self.A = A
        self.T = T
        self.R = R
        self.gamma = gamma
        self.tau = tau
        self.values = np.zeros((1, len(S)))
        # Where we'll store the softmaxed probabilities
        self.policy = np.zeros((len(A), len(S)))

    def ValueIteration(self, epsilon=0.00001):
        """
        Perform value iteration on MDP.

        Calculates each state's value and saves them in MDP's values attribute

        Args:
            epsilon (float): Convergence parameter

        Returns:
            None
        """
        self.values = np.zeros(self.values.shape)
        while True:
            V2 = self.values.copy()
            for i in range(0, len(self.S)):
                self.values[0, i] = (self.R[i] + self.gamma * (np.mat(self.T[i, :, :])) * np.mat(V2.transpose())).max()
            if (self.values - V2).max() <= epsilon:
                break

    def BuildPolicy(self, Softmax=True):
        """
        Build optimal policy

        Calculates MDPs optimal policy

        Args:
            Softmax (bool): Indicates if actions are softmaxed.

        Returns:
            None
        """
        # Build a policy using the results from value iteration
        for i in range(0, len(self.S)):
            options = np.mat(
                self.T[i, range(0, 4), :]) * np.mat(self.values.transpose()) + self.R[i]
            options = np.concatenate([options])
            # Prevent softmax from overflowing
            options = options - abs(max(options))
            # Softmax the policy
            if Softmax:
                try:
                    options = [math.exp(options[j] / self.tau)
                               for j in range(len(options))]
                except OverflowError:
                    print "WARNING: Failed to softmax policy"
                    raise
                # If all actions have no value then set a uniform distribution
                if sum(options) == 0:
                    self.policy[:, i] = [
                        1.0 / len(options) for j in range(len(options))]
                else:
                    self.policy[:, i] = [
                        options[j] / sum(options) for j in range(len(options))]
            else:
                totalchoices = sum([options[optloop] == max(options)
                                    for optloop in range(len(options))])
                self.policy[:, i] = [(1.0 / totalchoices if options[optloop] == max(options) else 0)
                                     for optloop in range(len(options))]

    def GetStates(self, StartingPoint, ActionSequence):
        """
        Produce the sequence of states with highest likelihood given a starting point and sequence of actions.

        Args:
            StartingPoint (int): State number where agent begins.
            ActionSequence (list): List of indices of actions.

        Returns:
            List of state numbers
        """
        StateSequence = [0] * (len(ActionSequence) + 1)
        StateSequence[0] = StartingPoint
        for i in range(len(ActionSequence)):
            StateSequence[i + 1] = (
                self.T[StateSequence[i], ActionSequence[i], :]).argmax()
        return StateSequence

    def Run(self, State, Simple=False):
        """
        Sample an action from the optimal policy given the state

        Args:
            State (int): State number where agent begins.
            Simple (bool): Some states have various actions all with an equally high value.
                           when this happens, Run will randomly select one of these actions.
                           if Simple is set to True, it will instead select the first highest-value action.

        Returns:
            List of state numbers
        """
        if Simple:
            ActSample = 0
        else:
            ActSample = random.uniform(0, 1)
        ActionProbs = self.policy[:, State]
        ActionChoice = -1
        for j in range(len(ActionProbs)):
            if ActSample < ActionProbs[j]:
                ActionChoice = j
                break
            else:
                ActSample -= ActionProbs[j]
        # Get next state
        EndStates = self.T[State, ActionChoice, :]
        StateSample = random.uniform(0, 1)
        for j in range(len(EndStates)):
            if StateSample < EndStates[j]:
                EndState = j
                break
            else:
                StateSample -= EndStates[j]
        return [EndState, ActionChoice]

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

# -*- coding: utf-8 -*-

"""
Markov Decision Process solver.
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import numpy as np
import math
import random


class MDP(object):

    def __init__(self, S=[], A=[], T=[], R=[], gamma=0.95, tau=0.01):
        """
        Markov Decision Process (MDP) class.

        Args:
            S (list): List of states
            A (list): List of actions
            T (matrix): Transition matrix where T[SO,A,SF] is the probability of moving from So to SF after taking action A
            R (matrix): Reward function where R[A,S] is the reward for taking action A in state S
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

    def ValueIteration(self, epsilon=0.0001):
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
                options = [(self.R[j, i] + self.gamma * (np.mat(self.T[i, :, :]))
                            * np.mat(V2.transpose())).max() for j in range(len(self.A))]
                self.values[0, i] = max(options)
            if (self.values - V2).max() <= epsilon:
                break

    def Validate(self):
        """
        Check that MDP object is correct.

        Args:
            None
        """
        print "Validating MDP..."
        dims = self.T.shape
        states = len(self.S)
        actions = len(self.A)
        if (dims[0] != dims[2]):
            print "ERROR: Transition matrix is not square. MDP-001"
            return 0
        if (states != dims[0]):
            print "ERROR: Transition matrix does not match number of states. MDP-002"
            return 0
        if self.S != range(states):
            print "ERROR: States are not correctly numbered. MDP-003"
            return 0
        if self.A != range(actions):
            print "ERROR: Actions are not correctly numbered. MDP-004"
            return 0
        if (dims[1] != actions):
            print "ERROR: Transition matrix does not match number of actions. MDP-005"
            return 0
        if (self.gamma >= 1) or (self.gamma <= 0):
            print "ERROR: Invalida value of gamma. MDP-006"
            return 0
        if (self.tau <= 0):
            if (self.tau is not None):
                print "ERROR: Invalida value of tau. MDP-009"
                return 0
        # Check that every vector adds up to 1
        res = (np.ndarray.flatten(np.sum(self.T, axis=2)) == 1)
        if len(res) != sum(res):
            print "ERROR: Transition matrix rows do not add up to 1. MDP-007"
            return 0
        return 1

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
                self.T[i, :, :]) * np.mat(self.values.transpose())
            options = options.tolist()
            # Prevent softmax from overflowing
            maxval = abs(max(options)[0])
            options = [options[j][0] - maxval for j in range(len(options))]
            # Softmax the policy
            if Softmax:
                try:
                    options = [math.exp(options[j] / self.tau)
                               for j in range(len(options))]
                except OverflowError:
                    print "ERROR: Failed to softmax policy. MDP-008"
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

    def Run(self, State, Softmax=False, Simple=False):
        """
        Sample an action from the optimal policy given the state.
        Note that if softmax is set to true then Simple is ignored (see below).

        Args:
            State (int): State number where agent begins.
            Softmax (bool): Simulate with softmaxed policy?
            Simple (bool): Some states have various actions all with an equally high value.
                           when this happens, Run() randomly selects one of these actions.
                           if Simple is set to True, it selects the first highest-value action.

        Returns:
            List of state numbers
        """
        if Softmax:
            # If softmaxing then select a random sample
            ActSample = random.uniform(0, 1)
            ActionProbs = self.policy[:, State]
            ActionChoice = -1
            for j in range(len(ActionProbs)):
                if ActSample < ActionProbs[j]:
                    ActionChoice = j
                    break
                else:
                    ActSample -= ActionProbs[j]
        else:
            maxval = max(self.policy[:, State])
            maxindices = [
                i for i, j in enumerate(self.policy[:, State]) if j == maxval]
            if Simple:
                ActionChoice = maxindices[0]
            else:
                ActionChoice = random.choice(maxindices)
        # Now find the next state
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

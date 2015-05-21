# -*- coding: utf-8 -*-

"""
PosteriorContainer wraps samples from the planner
and comes with supporting functions to analyze
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os.path
import math


class PosteriorContainer(object):

    """
    Class to handle posterior samples.
    """

    def __init__(self, C, R, L, ActionSequence, Planner=[]):
        """
        Create an object that stores sampling results.

        Args:
            C (list): List of cost samples
            R (list): List of reward samples
            L (list): List of log-likelihoods
            ActionSequence (list): List of actions
            Planner (Planner): (optional) Planner object
        """
        self.CostSamples = C
        self.RewardSamples = R
        self.LogLikelihoods = L
        self.CostDimensions = self.CostSamples.shape[1]
        self.RewardDimensions = self.RewardSamples.shape[1]
        self.Samples = self.RewardSamples.shape[0]
        self.Actions = ActionSequence
        self.MapFile = None
        if Planner is not None:
            self.CostNames = Planner.Map.StateNames
            self.StartingPoint = Planner.Map.StartingPoint
            self.ActionNames = Planner.Map.ActionNames
            self.ObjectLocations = Planner.Map.ObjectLocations
            self.ObjectTypes = Planner.Map.ObjectTypes
            self.ObjectNames = Planner.Map.ObjectNames
            self.SoftChoice = Planner.Agent.SoftmaxChoice
            self.SoftAction = Planner.Agent.SoftmaxAction
            self.actionTau = Planner.Agent.actionTau
            self.choiceTau = Planner.Agent.choiceTau
        else:
            self.CostNames = None
            self.StartingPoint = None
            self.ActionNames = None
            self.ObjectLocations = None
            self.ObjectTypes = None
            self.ObjectNames = None
            self.SoftChoice = None
            self.SoftAction = None
            self.actionTau = None
            self.choiceTau = None

    def AssociateMap(self, MapName):
        """
        Add a map name of Posterior PosteriorContainer. Function also checks if Map exists in library.

        Args:
            MapName (string): Name of map to use.
        """
        self.MapFile = MapName
        FilePath = os.path.dirname(__file__) + "/Maps/" + MapName + ".ini"
        if not os.path.isfile(FilePath):
            print "WARNING: PosteriorContainer is linked with a map that doesn't exist in Bishop's library."

    def LongSummary(self):
        """
        NEEDS UPDATE
        """
        self.Summary()
        self.AnalyzeConvergence()
        # 10 is the default input. Just sending it to avoid the print message
        self.PlotCostPosterior(10)
        self.PlotRewardPosterior(10)

    def CompareRewards(self):
        """
        Create a matrix where (i,j) is the probability that object i has a
        higher or equal reward than object j.
        """
        RewardComparison = np.zeros((self.RewardDimensions, self.RewardDimensions))
        for i in range(self.RewardDimensions):
            for j in range(i, self.RewardDimensions):
                for s in range(self.Samples):
                    if (self.RewardSamples[s, i] >= self.RewardSamples[s, j]):
                        RewardComparison[i][j] += np.exp(self.LogLikelihoods[s])
                    else:
                        RewardComparison[j][i] += np.exp(self.LogLikelihoods[s])
        return RewardComparison

    def CompareCosts(self):
        """
        Create a matrix where (i,j) is the probability that terrain i has a
        higher or equal cost than terrain j.
        """
        CostComparison = np.zeros((self.CostDimensions, self.CostDimensions))
        for i in range(self.CostDimensions):
            for j in range(i, self.CostDimensions):
                for s in range(self.Samples):
                    if (self.CostSamples[s, i] >= self.CostSamples[s, j]):
                        CostComparison[i][j] += np.exp(self.LogLikelihoods[s])
                    else:
                        CostComparison[j][i] += np.exp(self.LogLikelihoods[s])
        return CostComparison

    def GetExpectedCosts(self, limit=None):
        """
        Calculate the expected costs using the first N samples.

        Args:
            limit (int): Number of samples to use
        """
        ExpectedCosts = []
        if limit is None:
            limit = self.Samples - 1
        for i in range(self.CostDimensions):
            NL = np.exp(self.LogLikelihoods[0:(limit + 1)])
            if sum(NL) == 0:
                print "WARNING: All likelihoods are zero up to this point. POSTERIORCONTAINER-001"
                NL = [1.0 / NL.shape[0]]
            else:
                NL = NL / sum(NL)
            a = self.CostSamples[0:(limit + 1), i]
            b = NL
            res = sum([float(a[i])*float(b[i]) for i in range(limit + 1)])
            ExpectedCosts.append(res)
        return ExpectedCosts

    def GetExpectedRewards(self, limit=None):
        """
        Calculate the expected rewards using the first N samples.

        Args:
            limit (int): Number of samples to use
        """
        ExpectedRewards = []
        if limit is None:
            limit = self.Samples - 1
        for i in range(self.RewardDimensions):
            NL = np.exp(self.LogLikelihoods[0:(limit + 1)])
            if sum(NL) == 0:
                print "WARNING: All likelihoods are zero up to this point. POSTERIORCONTAINER-001"
                NL = [1.0 / NL.shape[0]]
            else:
                NL = NL / sum(NL)
            a = self.RewardSamples[0:(limit + 1), i]
            b = NL
            res = sum([float(a[i])*float(b[i]) for i in range(limit + 1)])
            ExpectedRewards.append(res)
        return ExpectedRewards

    def PlotCostPosterior(self, bins=None):
        """
        NEEDS UPDATE
        """
        if bins is None:
            print "Number of bins not specified. Defaulting to 10."
            bins = 10
        maxval = np.amax(self.CostSamples)
        f, axarr = plt.subplots(self.CostDimensions, sharex=True)
        binwidth = maxval * 1.0 / bins + 0.00001
        xvals = [binwidth * (i + 0.5) for i in range(bins)]
        for i in range(self.CostDimensions):
            yvals = [0] * bins
            insert_indices = [int(math.floor(j / binwidth))
                              for j in self.CostSamples[:, i]]
            for j in range(self.Samples):
                yvals[insert_indices[j]] += self.Likelihoods[j]
            axarr[i].plot(xvals, yvals, 'b-')
            if self.CostNames != None:
                axarr[i].set_title(self.CostNames[i])
        plt.show()

    def PlotRewardPosterior(self, bins=None):
        """
        NEEDS UPDATE
        """
        if bins == None:
            print "Number of bins not specified. Defaulting to 10."
            bins = 10
        maxval = np.amax(self.RewardSamples)
        f, axarr = plt.subplots(self.RewardDimensions, sharex=True)
        binwidth = maxval * 1.0 / bins + 0.00001
        xvals = [binwidth * (i + 0.5) for i in range(bins)]
        for i in range(self.RewardDimensions):
            yvals = [0] * bins
            insert_indices = [int(math.floor(j / binwidth))
                              for j in self.RewardSamples[:, i]]
            for j in range(self.Samples):
                yvals[insert_indices[j]] += self.Likelihoods[j]
            axarr[i].plot(xvals, yvals, 'b-')
        axarr[0].set_title("Target A")
        axarr[1].set_title("Target B")
        plt.show()

    def Summary(self, human=True):
        """
        NEEDS UPDATE
        """
        ExpectedRewards = self.GetExpectedRewards()
        RewardComp = self.CompareRewards()
        ObjAPred = self.ObjectAPrediction()
        ObjBPred = self.ObjectBPrediction()
        ExpectedCosts = self.GetExpectedCosts()
        CostMatrix = self.CompareCosts()
        # Combine all function to print summary here
        if human:
            sys.stdout.write("Map: " + str(self.MapFile) + "\n")
            sys.stdout.write(
                "To see map details run Bishop.LoadObserver(self).\n")
            sys.stdout.write("Targets: " + str(self.Targets) + "\n")
            sys.stdout.write(
                "Results using " + str(self.Samples) + " samples.\n")
            sys.stdout.write("\nPATH INFORMATION\n\n")
            sys.stdout.write(
                "Starting position: " + str(self.StartingCoordinates) + "\n")
            sys.stdout.write("Actions: " +
                             str(self.ActionNames) + " with")
            if self.Softmax:
                sys.stdout.write("out")
            sys.stdout.write(" softmaxed inference.\n")
            sys.stdout.write

            sys.stdout.write("\nGOAL PREDICTIONS\n\n")
            sys.stdout.write(
                "Probability that agent will get target A: " + str(ObjAPred) + "\n")
            sys.stdout.write(
                "Probability that agent will get target B: " + str(ObjBPred) + "\n")
            sys.stdout.write("\nINFERRED REWARDS\n\n")
            sys.stdout.write("Target A: " + str(ExpectedRewards[0]) + "\n")
            sys.stdout.write("Target B: " + str(ExpectedRewards[1]) + "\n")
            sys.stdout.write(
                "Probability that R(A)>R(B): " + str(RewardComp) + "\n")
            sys.stdout.write("\nINFERRED COSTS\n\n")
            if (self.CostNames != None):
                for i in range(self.CostDimensions):
                    sys.stdout.write(
                        str(self.CostNames[i]) + ": " + str(ExpectedCosts[i]) + "\n")
                sys.stdout.write(str(self.CostNames) + "\n")
            else:
                sys.stdout.write(str(ExpectedCosts) + "\n")
            sys.stdout.write(
                "Cost comparison matrix: i, j = p( C(i)>=C(j) )\n")
            sys.stdout.write(str(CostMatrix) + "\n")
        else:
            sys.stdout.write("WARNING: Printed limited version\n")
            sys.stdout.write(
                "Samples,StartingCoordinates,Actions,ObjectA,ObjectB,AvsB,PredictionA,PredictionB\n")
            sys.stdout.write(str(self.Samples) + "," + str(self.StartingCoordinates) +
                             "," + str(self.Actions) + "," +
                             str(ExpectedRewards[0]) + "," +
                             str(ExpectedRewards[1]) + "," + str(RewardComp) + "," + str(ObjAPred) + "," + str(ObjBPred) + "\n")

    def AnalyzeConvergence(self, jump=None):
        """
        NEEDS UPDATE
        """
        # jump indicates how often to recompute the average
        if jump is None:
            if self.Samples > 100:
                print "Recomputing average after every 20 samples"
                jump = int(round(self.Samples * 1.0 / 20))
            else:
                print "Recomputing average after every sample"
                jump = 1
        rangevals = range(0, self.Samples, jump)
        ycostvals = [self.GetExpectedCosts(i) for i in rangevals]
        ycostvals = np.array(ycostvals)
        yrewardvals = [self.GetExpectedRewards(i) for i in rangevals]
        yrewardvals = np.array(yrewardvals)
        # break it into plots.
        # Costs
        f, axarr = plt.subplots(self.CostDimensions, 2)
        for i in range(self.CostDimensions):
            axarr[i, 0].plot(rangevals, ycostvals[:, i], 'b-')
            if self.CostNames is not None:
                axarr[i, 0].set_title(self.CostNames[i])
        # Rewards
        axarr[0, 1].plot(rangevals, yrewardvals[:, 0], 'b-')
        axarr[0, 1].set_title("Target A")
        axarr[1, 1].plot(rangevals, yrewardvals[:, 1], 'b-')
        axarr[1, 1].set_title("Target B")
        plt.show()

    def SaveSamples(self, Name):
        """
        Save object as a pickle file.

        Args:
            Name (string): Filename. Function adds ".p" extension if it's not provided
        """
        if Name[-2:] != ".p":
            Name = Name + ".p"
        pickle.dump(self, open(Name, "wb"))

    def Display(self, Full=False):
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

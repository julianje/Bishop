import numpy as np
import matplotlib.pyplot as plt

class PosteriorContainer(object):
	"""
	Class to handle posterior samples.
	"""

	def __init__(self,C,R,L,A,B):
		self.CostSamples = C
		self.RewardSamples = R
		self.Likelihoods = L
		self.ObjectAOutcome = A
		self.ObjectBOutcome = B
		self.CostDimensions = self.CostSamples.shape[1]
		self.RewardDimensions = self.RewardSamples.shape[1]
		self.CostNames = None

	def ObjectAPrediction(self):
		return sum(self.ObjectAOutcome*self.Likelihoods)

	def AddCostNames(self,CostNames):
		self.CostNames=CostNames

	def ObjectBPrediction(self):
		return sum(self.ObjectBOutcome*self.Likelihoods)

	def GetExpectedRewards(self):
		ExpectedRewards=[]
		for i in range(self.RewardDimensions):
			ExpectedRewards.append(np.dot(self.RewardSamples[:, i], self.Likelihoods))
		return ExpectedRewards

	def GetExpectedCosts(self):
		ExpectedCosts=[]
		for i in range(self.CostDimensions):
			ExpectedCosts.append(np.dot(self.CostSamples[:, i], self.Likelihoods))
		return ExpectedCosts

	def PlotCostPosterior(self):
		f, axarr = plt.subplots(self.CostDimensions, sharex=True)
		for i in range(self.CostDimensions):
			axarr[i].scatter(self.CostSamples[:,i],self.Likelihoods,alpha=0.75)
			if self.CostNames!=None:
				axarr[i].set_title(self.CostNames[i])
		plt.show()

	def Display(self, Full=False):
		# Print class properties
		if Full:
			for (property, value) in vars(self).iteritems():
				print property, ': ', value
		else:
			for (property, value) in vars(self).iteritems():
				print property
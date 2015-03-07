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

	def __init__(self,C,R,L,A,B):
		self.CostSamples = C
		self.RewardSamples = R
		self.Likelihoods = L
		self.ObjectAOutcome = A
		self.ObjectBOutcome = B
		self.CostDimensions = self.CostSamples.shape[1]
		self.RewardDimensions = self.RewardSamples.shape[1]
		self.Samples = self.RewardSamples.shape[0]
		self.CostNames = None
		self.MapFile = None

	def AssociateMap(self,MapName):
		self.MapFile=MapName
		FilePath = os.path.dirname(__file__) + "/Maps/"+MapName+".ini"
		if not os.path.isfile(FilePath):
			print "WARNING: Map not found in library"

	def ObjectAPrediction(self):
		return sum(self.ObjectAOutcome*self.Likelihoods)

	def AddCostNames(self,CostNames):
		self.CostNames=CostNames

	def ObjectBPrediction(self):
		return sum(self.ObjectBOutcome*self.Likelihoods)

	def CompareRewards(self):
		probrws=0
		for i in range(self.Samples):
			if (self.RewardSamples[i,0]>=self.RewardSamples[i,1]):
				probrws+=self.Likelihoods[i]
		return probrws

	def CompareCosts(self):
		CostComparison = np.zeros((self.CostDimensions,self.CostDimensions))
		for i in range(self.CostDimensions):
			for j in range(i,self.CostDimensions):
				for s in range(self.Samples):
					if (self.CostSamples[s,i]>=self.CostSamples[s,j]):
						CostComparison[i][j]+=self.Likelihoods[s]
					else:
						CostComparison[j][i]+=self.Likelihoods[s]
		return CostComparison

	def GetExpectedCosts(self,limit=None):
		ExpectedCosts=[]
		if limit==None:
			limit = self.Samples-1
		for i in range(self.CostDimensions):
			NL = self.Likelihoods[0:(limit+1)]
			if sum(NL)==0:
				NL = [1.0/NL.shape[0]] * NL.shape[0]
			else:
				NL = NL/sum(NL)
			ExpectedCosts.append(np.dot(self.CostSamples[0:(limit+1), i], NL))
		return ExpectedCosts

	def GetExpectedRewards(self,limit=None):
		ExpectedRewards=[]
		if limit==None:
			limit = self.Samples-1
		for i in range(self.RewardDimensions):
			NL = self.Likelihoods[0:(limit+1)]
			if sum(NL)==0:
				NL = [1.0/NL.shape[0]] * NL.shape[0]
			else:
				NL = NL/sum(NL)
			ExpectedRewards.append(np.dot(self.RewardSamples[0:(limit+1), i], NL))
		return ExpectedRewards

	def PlotCostPosterior(self,bins=None):
		if bins == None:
			print "Number of bins not specified. Defaulting to 10."
			bins = 10
		maxval = np.amax(self.CostSamples)
		f, axarr = plt.subplots(self.CostDimensions, sharex=True)
		binwidth = maxval*1.0/bins+0.00001
		xvals = [binwidth*(i+0.5) for i in range(bins)]
		for i in range(self.CostDimensions):
			yvals = [0] * bins
			insert_indices = [int(math.floor(j/binwidth)) for j in self.CostSamples[:,i]]
			for j in range(self.Samples):
				yvals[insert_indices[j]]+=self.Likelihoods[j]
			axarr[i].plot(xvals,yvals, 'b-')
			if self.CostNames!=None:
				axarr[i].set_title(self.CostNames[i])
		plt.show()

	def PlotRewardPosterior(self,bins=None):
		if bins == None:
			print "Number of bins not specified. Defaulting to 10."
			bins = 10
		maxval = np.amax(self.RewardSamples)
		f, axarr = plt.subplots(self.RewardDimensions, sharex=True)
		binwidth = maxval*1.0/bins+0.00001
		xvals = [binwidth*(i+0.5) for i in range(bins)]
		for i in range(self.RewardDimensions):
			yvals = [0] * bins
			insert_indices = [int(math.floor(j/binwidth)) for j in self.RewardSamples[:,i]]
			for j in range(self.Samples):
				yvals[insert_indices[j]]+=self.Likelihoods[j]
			axarr[i].plot(xvals,yvals, 'b-')
		axarr[0].set_title("Target A")
		axarr[1].set_title("Target B")
		plt.show()

	def Summary(self, human=True):
		ExpectedRewards = self.GetExpectedRewards()
		RewardComp = self.CompareRewards()
		ObjAPred = self.ObjectAPrediction()
		ObjBPred = self.ObjectBPrediction()
		ExpectedCosts = self.GetExpectedCosts()
		CostMatrix=self.CompareCosts()
		# Combine all function to print summary here
		if human:
			sys.stdout.write("Results using "+str(self.Samples)+ " samples.\n")
			sys.stdout.write("\nINFERRED REWARDS\n\n")
			sys.stdout.write("Target A: "+str(ExpectedRewards[0])+"\n")
			sys.stdout.write("Target B: "+str(ExpectedRewards[1])+"\n")
			sys.stdout.write("Probability that R(A)>R(B): "+ str(RewardComp)+ "\n")
			sys.stdout.write("\nGOAL PREDICTIONS\n\n")
			sys.stdout.write("Probability that agent will get target A: "+ str(ObjAPred)+"\n")
			sys.stdout.write("Probability that agent will get target B: "+ str(ObjBPred)+ "\n")
			sys.stdout.write("\nINFERRED COSTS\n\n")
			if (self.CostNames!=None):
				for i in range(self.CostDimensions):
					sys.stdout.write(str(self.CostNames[i])+": "+str(ExpectedCosts[i])+"\n")
				sys.stdout.write(str(self.CostNames)+"\n")
			else:
				sys.stdout.write(str(ExpectedCosts)+"\n")
				sys.stdout.write("Cost comparison matrix (i,j = prob that C(terrain_i)>=C(terrain_j)):\n")
			sys.stdout.write(str(CostMatrix)+"\n")
		else:
			sys.stdout.write("WARNING: Printed limited version\n")
			sys.stdout.write("Samples,ObjectA,ObjectB,AvsB,PredictionA,PredictionB\n")
			sys.stdout.write(str(self.Samples)+","+str(ExpectedRewards[0])+","+
				str(ExpectedRewards[1])+","+str(RewardComp)+","+str(ObjAPred)+","+str(ObjBPred)+"\n")

	def AnalyzeConvergence(self,jump=1):
		# jump indicates how often to recompute the average
		xvals=range(self.Samples)
		rangevals=range(0,self.Samples,jump)
		ycostvals=[self.GetExpectedCosts(i) for i in rangevals]
		ycostvals=np.array(ycostvals)
		yrewardvals=[self.GetExpectedRewards(i) for i in rangevals]
		yrewardvals=np.array(yrewardvals)
		# break it into plots.
		# Costs
		f, axarr = plt.subplots(self.CostDimensions, 2)
		for i in range(self.CostDimensions):
			axarr[i,0].plot(xvals,ycostvals[:,i], 'b-')
			if self.CostNames!=None:
				axarr[i,0].set_title(self.CostNames[i])
		# Rewards
		axarr[0,1].plot(xvals,yrewardvals[:,0], 'b-')
		axarr[0,1].set_title("Target A")
		axarr[1,1].plot(xvals,yrewardvals[:,1], 'b-')
		axarr[1,1].set_title("Target B")
		plt.show()

	def SaveSamples(self, Name):
		FileName = Name + ".p"
		pickle.dump(self, open(FileName, "wb"))

	def Display(self, Full=False):
		# Print class properties
		if Full:
			for (property, value) in vars(self).iteritems():
				print property, ': ', value
		else:
			for (property, value) in vars(self).iteritems():
				print property
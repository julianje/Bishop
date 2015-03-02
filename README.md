#Bishop
______

## About

Bishop, after [Washington Bishop](http://en.wikipedia.org/wiki/Washington_Irving_Bishop), is a cognitive model of [Theory of mind](http://en.wikipedia.org/wiki/Theory_of_mind).

## Install and uninstall

	python setup.py install
	pip uninstall Bishop

## Running Bishop

#### Simulate agents traveling in an existing map

	import Bishop	
	Observer = Bishop.LoadEnvironment("Tatik_T1")
	Observer.SimulateAgents(StartingPoint=[6,6],Samples=100,HumanReadable=True)



#### Cost-reward inference given observable actions

	import Bishop
	Observer = Bishop.LoadEnvironment("Tatik_T1")
	ObservedPath = Observer.GetActionList(['L','L','U'])
	Res = Observer.InferAgent(StartingPoint=[6,6], ObservedPath, Samples=100, Softmax=True)

The result is a __PosteriorContainer__ object. Here are some things you can do with the output

	Res.Summary # Human-readable summary
	Res.AnalyzeConvergence() # Visually check if sampling converged
	Res.PlotCostPosterior()
	Res.PlotRewardPosterior()
	Res.SaveSamples("MyResults")

You can reload the samples and the observer model later with

	Res = Bishop.LoadSamples("MyResults.p")
	Observer = Bishop.LoadObserver(Res)

## Creating a new map

#### Through configuration files

A map consists of two files: An ASCII description, and a .ini configuration file.

#### Inside python

To generate a simple grid-world with one terrain start with

	MyMap = Bishop.Map()
	MyMap.BuildGridWorld(5,3,Diagonal=True)

This creates a 5 by 3 map that can be navigated diagonally. Terrain type is stored in MyMap.StateTypes

## How it works

Bishop does bayesian inference over optimal planners to infer the cost and rewards underlying an agent's actions. It then uses the posterior distribution of costs and rewards to predict how the agent will navigate.

## Details

Bishop has six classes. You can see what each class is saving by calling the Display() method on an object.

__Observer__ objects are rational observers. They require a Map and an Agent object (see below) and have three main methods:

* **ComputeLikelihood:** Computes the likelihood that the agent would take a given sequence of actions
* **SimulateAgent:** Simulates the agent from a given starting point
* **InferAgent:** Infers the agents costs and rewards given a sequence of observable actions

See Example folder for examples on how to use these methods.

__PosteriorContainer__ objects save cost and reward samples along with their likelihoods. They store some additional information and have methods for deriving meaningful results from the samples.

__Planner__ objects contain a Markov Decision Process and supporting methods for modifying the MDP structures.

__Map__ objects contained a map's description for the agent and the observer to use. It's main method is BuildGridWorld() to generate simple 2-dimensional grid worlds.

__Agent__ objects contain agent information.

__MDP__ objects store Markov Decision Processes along with exact planning algorithms and supporting methods.
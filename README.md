#Bishop
______

## About

Bishop, after [Washington Bishop](http://en.wikipedia.org/wiki/Washington_Irving_Bishop), is a cognitive model of [Theory of mind](http://en.wikipedia.org/wiki/Theory_of_mind).

## Installation

<code>python setup.py install</code>

Use pip to uninstall

<code>pip uninstall Bishop</code>

## Running Bishop

#### Simulate agents traveling in an existing map

<code>
	import Bishop
	
	Observer = Bishop.LoadEnvironment("Tatik_T1")
	
	Observer.SimulateAgents(StartingPoint=[6,6],Samples=100,HumanReadable=True)
</code>



#### Infer an agent's costs and rewards given their actions

<code>
	import Bishop

	Observer = Bishop.LoadEnvironment("Tatik_T1")
	
	ObservedPath = Observer.GetActionList(['L','L','U'])
	
	Res = Observer.InferAgent(StartingPoint=[6,6], ObservedPath, Samples=100, Softmax=True)
</code>

Res will be a PosteriorContainer object. Here are a couple of things you can do with Res

<code>
	Res.Summary # Human-readable summary

	Res.AnalyzeConvergence() # Visual check if sampling converged
	
	Res.PlotCostPosterior()
	
	Res.PlotRewardPosterior()
	
	Res.SaveSamples("MyResults")
</code>

You can then re-load the samples and the observer model...

<code>
	Res = Bishop.LoadSamples("MyResults.p")
	
	Observer = Bishop.LoadObserver(Res)
</code>

#### Creating a new map

##### Within python

##### Through configuraton files.

The first Map is an ASCII

Maps need two files. 

## How it works

Bishop has five kinds of objects you can build. You can see the contents of any object by calling their Display() method.

#### Observer

Observer objects are rational observers. They require a Map and an Agent (see below). Observers have three main methods:

* **ComputeLikelihood:** Computes the likelihood that the agent would take a given sequence of actions
* **SimulateAgent:** Simulates the agent from a given starting point
* **InferAgent:** Infers the agents costs and rewards given a sequence of observable actions

#### Planner

Planner objects contain an MDP and supporting methods. This is the main object the Observers use to infer agents.

#### Map

This object contains a map specification for the observer. It's main method is BuildGridWorld(xdimension, ydimension, Diagonal=[True/False]). The last parameter determines if traveling across the diagonals is allowed.

#### Agent

This object contains agent information. It's main method, ResampleAgent() re-generates a new random agent.

#### MDP

A Markov Decision Process object.
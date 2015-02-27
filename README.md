#Bishop
______

## About

Bishop, after [Washington Bishop](http://en.wikipedia.org/wiki/Washington_Irving_Bishop), is a cognitive model of [Theory of mind](http://en.wikipedia.org/wiki/Theory_of_mind).

## Installation

<code>python setup.py install</code>

## How it works

Bishop has five kinds of objects you can build. You can see the contents of any object by calling their Display() method.

### Observer

Observer objects are rational observers. They require a Map and an Agent (see below). Observers have three main methods:

* **ComputeLikelihood:** Computes the likelihood that the agent would take a given sequence of actions
* **SimulateAgent:** Simulates the agent from a given starting point
* **InferAgent:** Infers the agents costs and rewards given a sequence of observable actions

### Planner

Planner objects contain an MDP and supporting methods. This is the main object the Observers use to infer agents.

### Map

This object contains a map specification for the observer. It's main method is BuildGridWorld(xdimension, ydimension, Diagonal=[True/False]). The last parameter determines if traveling across the diagonals is allowed.

### Agent

This object contains agent information. It's main method, ResampleAgent() re-generates a new random agent.

### MDP

A Markov Decision Process object.

## Running Bishop

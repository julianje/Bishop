# Path generator for a given map

from Bishop import *

Observer = LoadEnvironment("Tatik_T1_L1")
# Simulate different agents
Observer.SimulateAgents(Samples=100, HumanReadable=True)
# Simulate same agent over and over again (output only changes is agent is
# softmaxed)
Observer.SimulateAgents(Samples=100, HumanReadable=True, ResampleAgent=False)

# Call using all parameters:
# Simulate 100 agents, output raw action ids,
# resample agent after each simulation
# when agent can take two equally good actions
# select the first one.
Observer.SimulateAgents(
    Samples=100,
    HumanReadable=False,
    ResampleAgent=True,
    Simple=True)

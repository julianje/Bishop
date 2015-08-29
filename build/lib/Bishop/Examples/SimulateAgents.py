# Path generator for a given map

from Bishop import *

Observer = LoadObserver("Tatik_T1_L1")
# Simulate different agents
Simulations = Observer.SimulateAgents(Samples=100, HumanReadable=True)
# Simulate same agent over and over again (output only changes is agent is
# softmaxed)
Simulations = Observer.SimulateAgents(Samples=100, HumanReadable=True, ResampleAgent=False)

# See cost and reward samples and the actions.
Simulations.Costs
Simulations.Rewards
Simulations.Actions
# Save simulations onto a .csv file
Simulations.SaveCSV("MySimulations.csv", overwrite=False)

# Call using all parameters:
# Simulate 100 agents, output raw action ids,
# resample agent after each simulation
# when agent can take two equally good actions
# select the first one.
Simulations = Observer.SimulateAgents(
    Samples=100,
    HumanReadable=False,
    ResampleAgent=True,
    Simple=True)

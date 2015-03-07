# Path generator for a given map

import Bishop

Observer = Bishop.LoadEnvironment("Tatik_T1")
Observer.SimulateAgents(StartingCoordinates=[6,6],Samples=100,HumanReadable=True)

# Call using all parameters:
# Simulate 100 agents starting in 6,6,
# with softmax. Output raw action numbers
# Force first terrain (in this map mud)
# to always be the least costly terrain
Observer.SimulateAgents(
	StartingCoordinates=[6,6],
	HumanReadable=False,
	Samples=100,
	Simple=False,
	Softmax=True,
	ConstrainTerrains=True)
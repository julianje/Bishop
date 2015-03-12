# Bishop inference
import Bishop

MapName = "Tatik_T1"

Observer = Bishop.LoadEnvironment(MapName)

ObservedPath = Observer.GetActionList(['UL','R'])

Res = Observer.InferAgent(StartingCoordinates=[6,6],
	ActionSequence=ObservedPath,
	Samples=500,
	Softmax=True,
	CostRestriction=False)

# Associate a map name with the samples
Res.AssociateMap(MapName)
# Human-friendly summary
Res.Summary()
# Or print csv-style
Res.Summary(human=False)
# Visually assess if samples converged
Res.AnalyzeConvergence()
# Look at cost and reward posterior plots
Res.PlotCostPosterior()
Res.PlotRewardPosterior()
# Probability that agent will get Targets A and B
Res.ObjectAPrediction()
Res.ObjectBPrediction()
# Probability that R(A)>R(B)
Res.CompareRewards()
# Get expected costs and rewards
Res.GetExpectedCosts()
Res.GetExpectedRewards()
# Show cost comparison matrix
Res.CompareCosts()
# Do everything above at once
Res.LongSummary()
# Save results
Res.SaveSamples("MyResults")

# Load samples
Res = Bishop.LoadSamples("MyResults.p")
# Or just look at them
Bishop.AnalyzeSamples("MyResults.p") # Does not return samples
# Load the observer model associated with samples
Observer = Bishop.LoadObserver(Res) # Only works if Results had a mapp associated
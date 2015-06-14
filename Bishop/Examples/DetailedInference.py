# Bishop inference

from Bishop import *

MapName = "Tatik_T1"

Observer = Bishop.LoadEnvironment(MapName)

Res = Observer.InferAgent(
    ActionSequence=['UL', 'R'], Samples=500, Feedback=True)

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
SaveSamples(Res, "MyResults")

# Load samples
Res = LoadSamples("MyResults.p")
# Or just look at the long summary without returning the files to the workspace
AnalyzeSamples("MyResults.p")
# Load the observer model associated with samples
Observer = LoadObserver(Res)  # Only works if Results had a map associated

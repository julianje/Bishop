# Infer an agent's path

import Bishop

Observer = Bishop.LoadEnvironment("Tatik_T1")

Samples = 10
ObservedPath = Observer.M.GetActionList(['UL','UL','UL'])
StartingPoint=[6,6]

Res = Observer.InferAgent(StartingPoint, ObservedPath, Samples, Softmax=True)

# Probability that agent will pickup object A?
Res.ObjectAPrediction()
# Probability that agent will pickup object B?
Res.ObjectBPrediction()
# Infer costs
Res.GetExpectedCosts()
# Infer rewards
Res.GetExpectedRewards()
# Plot costs
Res.PlotCostPosterior()

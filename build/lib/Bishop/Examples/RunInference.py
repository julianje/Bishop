# Infer an agent's path

import Bishop

Observer = Bishop.LoadEnvironment("Tatik_T1")

Samples = 10
ObservedPath = Observer.M.GetActionList(['UL','UL','UL'])
StartingPoint=[6,6]

Res = Observer.InferAgent(StartingPoint, ObservedPath, Samples, Softmax=True)

Res.ObjectAPrediction()
Res.ObjectBPrediction()
Res.GetExpectedCosts()
Res.GetExpectedRewards()
Res.PlotCostPosterior()

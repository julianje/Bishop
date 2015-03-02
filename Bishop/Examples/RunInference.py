# Infer an agent's path

import Bishop

Observer = Bishop.LoadEnvironment("Tatik_T1")

Samples = 250
ObservedPath = Observer.M.GetActionList(['R'])
StartingPoint=[6,6]

Res = Observer.InferAgent(StartingPoint, ObservedPath, Samples, Softmax=True)

Res.Summary()

Res.AnalyzeConvergence()

Res.PlotRewardPosterior()
Res.PlotCostPosterior()
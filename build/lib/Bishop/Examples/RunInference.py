# Infer an agent's path

import Bishop

Observer = Bishop.LoadEnvironment("Tatik_T1")

Samples = 50
ObservedPath = Observer.M.GetActionList(['UL'])
StartingPoint=[6,6]

Res = Observer.InferAgent(StartingPoint, ObservedPath, Samples, Softmax=True)

Res.Summary()

Res.PlotCostPosterior()

Res.AnalyzeConvergence()
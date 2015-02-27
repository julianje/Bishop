# Infer an agent's path

import Bishop

Observer = Bishop.LoadEnvironment("Tatik_T1")

Samples = 100
StartingPoint = 45
ObservedPath = [4, 4, 4]

[Costs, Rewards, Likelihoods, APursuit, BPursuit] = Observer.InferAgent(
    StartingPoint, ObservedPath, Samples, Softmax=True)
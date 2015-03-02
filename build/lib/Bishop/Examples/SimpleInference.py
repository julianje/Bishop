# Bishop inference
import Bishop
Observer = Bishop.LoadEnvironment("Tatik_T1")
ObservedPath = Observer.M.GetActionList(['R'])
Res = Observer.InferAgent(StartingPoint=[6,6], ActionSequence=ObservedPath, Samples=50)
Res.Summary()
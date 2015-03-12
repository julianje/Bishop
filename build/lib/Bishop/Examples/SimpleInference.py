# Bishop inference
import Bishop
Observer = Bishop.LoadEnvironment("Tatik_T1")
ObservedPath = Observer.GetActionList(['R'])
Res = Observer.InferAgent(StartingCoordinates=[6,6], ActionSequence=ObservedPath, Samples=50)
Res.Summary()
# Bishop inference

from Bishop import *

Observer = LoadEnvironment("Tatik_T1_L1")
ObservedPath = Observer.GetActionList(['R', 'R'])
# When Feedback is True function prints percentage complete
Res = Observer.InferAgent(
    ActionSequence=ObservedPath, Samples=500, Feedback=True)
Res.Summary()

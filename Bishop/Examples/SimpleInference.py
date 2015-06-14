# Bishop inference

from Bishop import *

Observer = LoadEnvironment("Tatik_T1_L1")
# When Feedback is True function prints percentage complete
Res = Observer.InferAgent(
    ActionSequence=['R', 'R'], Samples=500, Feedback=True)
Res.Summary()

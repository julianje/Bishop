# -*- coding: utf-8 -*-

"""
Quick run through arguments.
Call this function and send Map name, samples,
and action sequence as arguments.

>> python QuickRun.py Tatik_T1_L1 100 2 2 2

runs inference on map Tatik_T1_L1 using 100 samples
and action sequence 2 2 2.
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

from Bishop import *
import sys

Path_length = len(sys.argv) - 3
ActionSequence = [int(s) for s in sys.argv[3:]]
O = LoadEnvironment(sys.argv[1], True)
Res = O.InferAgent(ActionSequence, int(sys.argv[2]))
Res.Summary(False)
SaveSamples(Res, str(sys.argv))

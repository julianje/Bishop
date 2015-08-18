# -*- coding: utf-8 -*-

"""
Script to quickly run cost-reward inference on an existing map.
Arguments (separated by space):
    Map[string]: Name of map (should exist on Bishop's library or on your local folder)
    Samples[int]: Number of samples on inference
    Actions[list]: List of integer codes for action sequence.

Example:

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
O = LoadMap(sys.argv[1], False, True)
Res = O.InferAgent(ActionSequence, int(sys.argv[2]))
Res.Summary(False)
SaveSamples(Res, str(sys.argv))

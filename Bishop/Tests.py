from Bishop import *
import numpy as np
# Test MDP.
# Create a one-dimension MDP manually
S = range(5)
A = range(2)  # left or right
R = [50, -1, -1, -1, 0]
T = np.zeros((5, 2, 5))
# Move left
T[0][0][0] = 1
T[1][0][0] = 1
T[2][0][1] = 1
T[3][0][3] = 1
T[4][0][4] = 1
# Move right
T[0][1][1] = 1
T[1][1][2] = 1
T[2][1][3] = 1
T[3][1][4] = 1
T[4][1][4] = 1
Test = MDP.MDP(S, A, T, R)
Test.validate()

from Bishop import *
import numpy as np

############
# Test MDP #
############

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

############
# Test Map #
############

# Create a simple grid world
Test = Map()
Test.BuildGridWorld(3, 4, diagonal=True)
Test.PrintMap()
Test.Validate()  # Should fail.
Test.InsertObjects([1, 2, 4], [0, 0, 1], ["A", "B"])
Test.AddStartingPoint(10)
Test.AddExitState(0)
Test.Validate()  # Success!

# Repeat above, but do not number object types correctly.
Test = Map()
Test.BuildGridWorld(3, 4, diagonal=False)
Test.Validate()  # Should fail.
Test.InsertObjects([1, 2, 4], [1, 1, 2], ["A"])  # Fail
Test.InsertObjects([1, 2, 4], [1, 1, 2], ["A", "B"])  # Ok
Test.AddStartingPoint(10)
Test.AddExitState(0)
Test.Validate()  # Should fail

# Repeat above, but using coordinate system
Test = Map()
Test.BuildGridWorld(3, 4, diagonal=False)
Test.PrintMap()
Test.Validate()  # Should fail.
S1 = Test.GetRawStateNumber([2, 1])
S2 = Test.GetRawStateNumber([3, 1])
S3 = Test.GetRawStateNumber([2, 2])
Test.InsertObjects([S1, S2, S3], [0, 0, 1], ["A", "B"])
Test.AddStartingPoint(10)
Test.AddExitState(0)
Test.Validate()  # Success!

##############
# Test Agent #
##############

MyMap = Map()
MyMap.BuildGridWorld(1, 3, diagonal=True)
MyMap.InsertObjects([0], [0], ["A"])
MyMap.AddStartingPoint(1)
MyMap.AddExitState(2)
MyMap.PrintMap()
MyAgent = Agent(MyMap, "ScaledUniform", 1, 30)

################
# Test Planner #
################

MyMap = Map()
MyMap.BuildGridWorld(1, 3, diagonal=True)
MyMap.InsertObjects([0], [0], ["A"])
MyMap.AddStartingPoint(1)
MyMap.AddExitState(2)
MyAgent = Agent(MyMap, "ScaledUniform", 1, 30)
MyPlanner = Planner.Planner(MyAgent, MyMap)
MyPlanner.Simulate()

#########################
# Test more complex map #
#########################

MyMap = Map()
MyMap.BuildGridWorld(4, 5, diagonal=True)
MyMap.InsertObjects([3, 16], [0, 1], ["A", "B"])
MyMap.InsertSquare(2, 1, 3, 3, 1)
MyMap.AddStartingPoint(19)
MyMap.AddExitState(0)
MyAgent = Agent(MyMap, "ScaledUniform", 1, 30)
MyPlanner = Planner.Planner(MyAgent, MyMap)
MyPlanner.Simulate()

#################
# Test observer #
#################

MyMap = Map()
MyMap.BuildGridWorld(4, 5, diagonal=True)
MyMap.InsertObjects([3, 16], [0, 1], ["A", "B"])
MyMap.InsertSquare(2, 1, 3, 3, 1)
MyMap.AddStartingPoint(19)
MyMap.AddExitState(0)
MyAgent = Agent(MyMap, "ScaledUniform", 1, 30)
Obs = Observer(MyAgent, MyMap)
Obs.SimulateAgents(10)
Obs.SimulateAgents(10,True)

















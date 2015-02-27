# Built for ComplexPlanner to manage.
# Map has a simpler representation of the environment (no exit states),
# allowing Planner to build the deep transition matrix.

# Structure is a little different. ComplexPlanner now handles how to build
# the MDP.

import numpy as np

class Map(object):

    """
    Map class.

    This class stores the environments states (S) and the terrain type (StateTypes), the possible actions (A), the transition matrix (T), and reward locations (Locations).
    It also stores human-readable information: StateNames, ActionNames, and LocationNames.

    Map comes with functions that help you build the transition matrix (BuildGridWorld).
    The Map specification is not the map we do planning on. Planner takes the Map description and uses it to build the true MDP (where objects can be removed).
    """

    def __init__(self, Locations=[[], [], [], []], S=[], StateTypes=[], StateNames=[], A=[], ActionNames=[], LocationNames=[], T=[]):
        """
        Create New map

        If no arguments are provided the structures are just initialized.

        ARGUMENTS:
        Locations[]      An array marking the locations of the targets.
                         There are four possible types of objects. Each entry
                         marks the locations of object 1, object 2, agent 1, and
                         agent 2, respectively.
        S                Set of states in the world.
        StateTypes[]     A set that matches in length the set of states and marks their
                         terrain type (As a discrete number).
        StateNames       A list containing names for each possible state type.
        A                Set of actions the world allows for
        ActionNames      Names of the actions
        LocationNames    List containing names of the objects taht are placed on the map.
                         LocationNames[i] contains the names of objects in Locations[i]
        T                Transition matrix. T[so,a,sf] contains the probability of switching
                         to state sf when taking action a in state so.
        """
        self.S = S
        self.A = A
        self.T = T
        self.ActionNames = ActionNames
        self.Locations = Locations
        self.LocationNames = LocationNames
        self.StateNames = StateNames
        self.StateTypes = StateTypes
        # Ensure rest of code breaks if BuildGridWorld wasn't called.
        self.x = -1
        self.y = -1

    def BuildGridWorld(self, x, y, Diagonal=False):
        """
        Build a simple grid world with a noiseless transition matrix.
        This gives the basic structure that can then be used to build the MDPs real transition matrix.

        ARGUMENTS
        x [integer]     Map's length
        y [integer]     Map's height
        """
        self.x = x
        self.y = y
        WorldSize = x * y
        self.S = range(WorldSize)
        self.StateTypes = [0] * len(self.S)
        if Diagonal:
            self.A = range(8)
            self.ActionNames = ["L", "R", "U", "D", "UL", "UR", "DL", "DR"]
        else:
            self.A = range(4)
            self.ActionNames = ["L", "R", "U", "D"]
        self.LocationNames = ["Object A", "Object B", "Agent A", "Agent B"]
        #From, With, To
        self.T = np.zeros((len(self.S), len(self.A), len(self.S)))
        # Make all states of the same type
        self.StateTypes = [0] * (len(self.S))
        for i in range(len(self.S)):
            # Moving left
            if (i % x == 0):
                self.T[i, 0, i] = 1
            else:
                self.T[i, 0, i - 1] = 1
            # Moving right
            if (i % x == x - 1):
                self.T[i, 1, i] = 1
            else:
                self.T[i, 1, i + 1] = 1
            # Moving up
            if (i < x):
                self.T[i, 2, i] = 1
            else:
                self.T[i, 2, i - x] = 1
            # Moving down
            if (i + x >= WorldSize):
                self.T[i, 3, i] = 1
            else:
                self.T[i, 3, i + x] = 1
            if Diagonal:  # Add diagonal transitions.
	            if ((i % x == 0) or (i < x)):  # Left and top edges
	                self.T[i, 4, i] = 1
	            else:
	                self.T[i, 4, i - x - 1] = 1
	            if ((i < x) or (i % x == x - 1)):  # Top and right edges
	                self.T[i, 5, i] = 1
	            else:
	                self.T[i, 5, i - x + 1] = 1
	            if ((i % x == 0) or (i + x >= WorldSize)):  # Bottom and left edges
	                self.T[i, 6, i] = 1
	            else:
	                self.T[i, 6, i + x - 1] = 1
	            # Bottom and right edges
	            if ((i % x == x - 1) or (i + x >= WorldSize)):
	                self.T[i, 7, i] = 1
	            else:
	                self.T[i, 7, i + x + 1] = 1

    def GetActionList(self, Actions):
        """
        GetActionList(ActionList)
        Transform a list of action names into the corresponding action numbers in the Map.
        This function helps make inference code human-readable
        """
        ActionList = [0] * len(Actions)
        for i in range(len(Actions)):
            ActionList[i] = self.ActionNames.index(Actions[i])
        return ActionList

    def GetWorldSize(self):
        """
        GetWorldSize()
        returns number of states.
        """
        return len(self.S)

    def NumberOfActions(self):
        """
        NumberOfActions()
        returns number of actions.
        """
        return len(self.A)

    def GetActionNames(self, Actions):
        """
        GetActionNames(Actions)
        Receives a list of action numbers and returns the names for the actions.
        """
        ActionNames = [0] * len(Actions)
        for i in range(len(Actions)):
            ActionNames[i] = self.ActionNames[Actions[i]]
        return ActionNames

    def InsertTargets(self, Locations):
        """
        InsertTargets(Locations)
        Adds objects to the map. Locations must be of the form [[],[],[],[]], with a total of two states.
        State numbers in the first list are objects of type 1. State numbers in the second list are objects of type 2.
        State numbers in the third and fourth lists are agents.

        Example:
        InsertTargets([0,1],[],[],[]) # Insert two objects of the same kind on states 0 and 1
        InsertTargets([0],[1],[],[]) # Insert two different objects on states 0 and 1
        InsertTargets([0],[],[1],[]) # Insert an object in state 0 and an agent who needs help in state 1
        InsertTargets([],[],[0,1],[]) # Insert two agents of the same kind on states 0 and 1. This means the algorithm will assume the motivation to save both agents is the same.
        InsertTargets([],[],[0],[1]) # Insert two different agents in states 0 and 1. This way, the agent might have different motivation for saving the different agents.
        """
        # Store the state position of the objects in the world.
        if sum(map(len, Locations)) > 2:
            print "Warning: More than two rewards on Map. Code will work, but this is a deviation from the experimental design."
        self.Locations = Locations

    def PullTargetStates(self, AddTerminalState=True):
        """
        PullTargetSTates(AddTerminalState=True)
        Returns a list of states that have an object in them. If AddTerminalState is set to True the list will include states where the agent can leave the Map.
        This function is useful for running simulations that can stop whenever something interesting happens (i.e., when the agent reaches an object, or when it leaves the map).
        """
        TargetStates = []
        for i in range(len(self.Locations)):
            discard = [TargetStates.append(j) for j in self.Locations[i]]
        if AddTerminalState:
            TargetStates.append(max(self.S))
        return TargetStates

    def AddStateNames(self, StateNames):
        """
        AddStateNames(StateNames) takes a list of the length of the terrain types giving them names.
        """
        self.StateNames = StateNames

    def Display(self, Full=False):
        """
        Print class attributes.
        Display() prints the stored values.
        Display(False) prints the variable names only.
        """
        if Full:
            for (property, value) in vars(self).iteritems():
                print property, ': ', value
        else:
            for (property, value) in vars(self).iteritems():
                print property

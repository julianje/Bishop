# -*- coding: utf-8 -*-

"""
Class to store Maps.
"""

__license__ = "MIT"

import numpy as np
import sys
import math


class Map(object):

    def __init__(self, Locations=[], LocationTypes=[], ObjectNames=[], S=[], StateTypes=[], StateNames=[], A=[], ActionNames=[], diagonal=None, T=[], ExitState=None, StartingPoint=None):
        """
        Create New map

        This class stores the environments states (S) and the terrain type (StateTypes), the possible actions (A), the transition matrix (T), and reward locations (Locations).
        It also stores human-readable information: StateNames, ActionNames, and LocationNames.
        If no arguments are provided the structures are just initialized.

        .. Warning::

           The constructor is designed to only initialize variables. Objects should be build through BuildGridWorld method.

        Args:
            Locations (list): List of object locations.
            LocationTypes (list): List indicating the object type in each location.
            ObjectNames (list): List with object names.
            S (list): List of states.
            StateTypes (list): List indicating the terrain type of each state.
            StateNames (list): List of strings indicating the names of states.
            A (list): List of actions available.
            ActionNames (list): List of action names.
            diagonal (boolean): Determines if agents can travel diagonally.
            T (matrix): Transition matrix. T[SO,A,SF] contains the probability that agent will go from SO to SF after taking action A.
            ExitState (int): Exit state
            StartingPoint (int): Map's starting point
        """
        self.diagonal = diagonal
        self.S = S
        self.A = A
        self.T = T
        self.ActionNames = ActionNames
        self.Locations = Locations
        self.LocationTypes = LocationTypes
        self.ObjectNames = ObjectNames
        self.StateNames = StateNames
        self.StateTypes = StateTypes
        self.ExitState = ExitState
        self.StartingPoint = StartingPoint
        # Ensure rest of code breaks if BuildGridWorld wasn't called.
        self.x = -1
        self.y = -1

    def Validate(self):
        """
        Check if Map object has everything it needs.
        """
        # Test 1. Check transition matrix has correct size.
        Tshape = self.T.shape
        if Tshape[0] != Tshape[2]:
            print "ERROR: Transition matrix has wrong dimensions"
            return 0
        if Tshape[0] != len(self.S):
            print "ERROR: Transition matrix does not match number of states"
            return 0
        if Tshape[1] != len(self.A):
            print "ERROR: Transition matrix does not match number of actions"
            return 0
        # Check that location and locationtype match
        if len(self.Locations) == 0 or len(self.LocationTypes) == 0:
            print "ERROR: Missing object locations"
            return 0
        if len(self.Locations) != len(self.LocationTypes):
            print "ERROR: List of locations and list of location types are of different length"
            return 0
        # Check that objectnames match number of objects
        if self.ObjectNames is not None:
            if len(self.ObjectNames) != len(set(self.Locations)):
                print "ERROR: Object names does not match number of objects"
                return 0
        # Check that starting point and exit state are in map
        if self.StartingPoint is not None:
            if self.StartingPoint < 0 or self.StartingPoint >= len(self.S):
                print "ERROR: Starting point is not a state number"
                return 0
        else:
            print "ERROR: Missing starting point."
            return 0
        if self.ExitState is not None:
            if self.ExitState < 0 or self.ExitState >= len(self.S):
                print "ERROR: Exit state is not a state number"
                return 0
        else:
            print "ERROR: Missing exit states."
            return 0
        # Check that transition matrix makes sense
        if sum([np.all(np.sum(self.T[:, i, :], axis=1) == 1) for i in range(len(self.A))]) != len(self.A):
            print "ERROR: Transition matrix is not well formed"
            return 0
        return 1

    def BuildGridWorld(self, x, y, diagonal=True):
        """
        Build a simple grid world with a noiseless transition matrix.

        Args:
            x (int): Map's length
            y (int): Map's height
            diagonal (bool): Can the agent travel diagonally?
        """
        self.x = x
        self.y = y
        self.diagonal = diagonal
        WorldSize = x * y
        self.S = range(WorldSize)
        self.StateTypes = [0] * len(self.S)
        if diagonal:
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
            if diagonal:  # Add diagonal transitions.
                if ((i % x == 0) or (i < x)):  # Left and top edges
                    self.T[i, 4, i] = 1
                else:
                    self.T[i, 4, i - x - 1] = 1
                if ((i < x) or (i % x == x - 1)):  # Top and right edges
                    self.T[i, 5, i] = 1
                else:
                    self.T[i, 5, i - x + 1] = 1
                # Bottom and left edges
                if ((i % x == 0) or (i + x >= WorldSize)):
                    self.T[i, 6, i] = 1
                else:
                    self.T[i, 6, i + x - 1] = 1
                # Bottom and right edges
                if ((i % x == x - 1) or (i + x >= WorldSize)):
                    self.T[i, 7, i] = 1
                else:
                    self.T[i, 7, i + x + 1] = 1

    def InsertSquare(self, topleftx, toplefty, width, height, value):
        """
        Insert a square of some type of terrain. This function rewrites old terrains

        Args:
            topleftx (int): x coordinate of top left corner of square (counting from left to right)
            toplefty (int): y coordinate of top left corner of square (counting from top to bottom)
            width (int): Square's width
            height (int): Square's height
        """

        if ((topleftx + width - 1) > self.x) or ((toplefty + height - 1) > self.y):
            print "ERROR: Square doesn't fit in map."
            return None
        TopLeftState = (toplefty - 1) * self.x + (topleftx) - 1
        for i in range(height):
            initial = TopLeftState + self.x * i
            end = TopLeftState + width + 1
            self.StateTypes[initial:end] = [value] * width

    def GetActionList(self, Actions):
        """
        Transform a list of action names into a list of action indices.

        Args:
            Actions (list): List of action names

        Returns
            Actions (list): List of actions in numerical value
        """
        ActionList = [0] * len(Actions)
        for i in range(len(Actions)):
            ActionList[i] = self.ActionNames.index(Actions[i])
        return ActionList

    def GetWorldSize(self):
        """
        Get size of world

        Args:
            None

        Returns:
            Size (int)
        """
        return len(self.S)

    def NumberOfActions(self):
        """
        Get number of actions

        Args:
            None

        Returns:
            Number of actions (int)
        """
        return len(self.A)

    def GetActionNames(self, Actions):
        """
        Get names of actions

        Args:
            Actions (list): List of actions in numerical value

        Returns
            Actions (list): List of action names
        """
        ActionNames = [0] * len(Actions)
        for i in range(len(Actions)):
            ActionNames[i] = self.ActionNames[Actions[i]]
        return ActionNames

    def GetRawStateNumber(self, Coordinates):
        """
        Transform coordinate into raw state number

        Args:
            Coordinates (list): with the x and y coordinate.

        Returns
            State (int)
        """
        # Transform coordinates to raw state numbers.
        xval = Coordinates[0]
        yval = Coordinates[1]
        return (yval - 1) * self.x + xval - 1

    def GetCoordinates(self, State):
        """
        Transform raw state number into coordinates

        Args:
            State (int): State id

        Returns
            Coordinates (list): x and y coordinates ([x,y])
        """
        yval = int(math.floor(State / self.x)) + 1
        xval = State - self.x * (yval - 1) + 1
        return [xval, yval]

    def InsertTargets(self, Locations, LocationTypes, ObjectNames=None):
        """
        Add objects to map.

        Args:
            Locations (list): List of state numbers where objects should be placed
            LocationTypes (list): List of identifiers about object id
            ObjectNames (list): List of names for the objects

        Returns:
            None

        Example: Add five objects on first five states. First two and last three objects are of the same kind, respectively.
        >> InsertTargets([0,1,2,3,4],[0,0,1,1,1],["Object A","Object B"])
        """
        # Check that location and locationtype match
        if len(Locations) != len(LocationTypes):
            print "ERROR: List of locations and list of location types are of different length"
            return None
        # Check that objectnames match number of objects
        if ObjectNames is not None:
            if len(ObjectNames) != len(set(Locations)):
                print "ERROR: Object names does not match number of objects"
                return None
        self.Locations = Locations
        self.LocationTypes = LocationTypes
        self.ObjectNames = ObjectNames

    def PullTargetStates(self, Coordinates=True):
        """
        PullTargetSTates(AddTerminalState=True)
        Returns a list of states that have an object in them.
        When Coordinates is set to false the function returns the raw state numbers
        """
        TargetStates = []
        for i in range(len(self.Locations)):
            discard = [TargetStates.append(j) for j in self.Locations[i]]
        if not Coordinates:
            return TargetStates
        else:
            return [self.GetCoordinates(item) for item in TargetStates]

    def AddStateNames(self, StateNames):
        """
        Add names to the states.

        Args:
            StateNames (list): List of strings with state names
        """
        self.StateNames = StateNames

    def AddExitState(self, ExitState):
        """
        Add exit state to map
        """
        if ExitState < 0 or ExitState >= len(self.S):
            print "ERROR: Exit state is not a state in the map."
            return None
        self.ExistState = ExitState

    def AddStartingPoint(self, StartingPoint):
        """
        Add starting point to map
        """
        if StartingPoint < 0 or StartingPoint >= len(self.S):
            print "ERROR: Starting point is not a state in the map."
            return None
        self.StartingPoint = StartingPoint

    def PrintMap(self):
        """
        Print map in ascii

        Args:
            None

        >> MyMap.PrintMap()
        """
        sys.stdout.write("Possible actions: " + str(self.ActionNames) + "\n")
        sys.stdout.write("Diagonal travel: " + str(self.diagonal) + "\n")
        sys.stdout.write("Targets: ")
        sys.stdout.write(str(self.PullTargetStates(True)) + "\n\n")
        print "Terrain types"
        for i in range(len(self.StateNames)):
            sys.stdout.write(self.StateNames[i] + ": " + str(i) + "\n")
        sys.stdout.write("\n")
        for i in range(self.y):
            for j in range(self.x):
                sys.stdout.write(str(self.StateTypes[self.x * i + j]))
            sys.stdout.write("\n")

    def Display(self, Full=False):
        """
        Print object attributes.

        .. Internal function::

           This function is for internal use only.

        Args:
            Full (bool): When set to False, function only prints attribute names. Otherwise, it also prints its values.

        Returns:
            standard output summary
        """
        if Full:
            for (property, value) in vars(self).iteritems():
                print property, ': ', value
        else:
            for (property, value) in vars(self).iteritems():
                print property

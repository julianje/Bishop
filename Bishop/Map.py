# -*- coding: utf-8 -*-

"""
Map class. Maps are a essentially an abstraction on top of MDPs that make it move intuitive to interact with the planner.
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import numpy as np
import sys
import math


class Map(object):

    def __init__(self, ObjectLocations=[], ObjectTypes=[], Organic=[], SurvivalProb=1, ObjectNames=[], S=[], StateTypes=[], StateNames=[], A=[], ActionNames=[], diagonal=None, T=[], ExitState=None, StartingPoint=None):
        """

        This class stores the environments states (S) and the terrain type (StateTypes), the possible actions (A), the transition matrix (T), and reward locations (ObjectLocations).
        It also stores human-readable information: StateNames, ActionNames, and LocationNames.
        If no arguments are provided the structures are just initialized.

        .. Warning::

           The constructor is designed to only initialize variables. Objects should be built through BuildGridWorld method.

        Args:
            ObjectLocations (list): List of object locations.
            ObjectTypes (list): List indicating the object type in each location.
            Organic (list): List indicating which object types are organic (i.e. might die at any point and thus have a future discount adjustment).
            SurvivalProb (float): Probability that organic objects survive in each time step.
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
        self.ObjectLocations = ObjectLocations
        self.ObjectTypes = ObjectTypes
        self.Organic = Organic
        self.SurvivalProb = SurvivalProb
        self.ObjectNames = ObjectNames
        self.StateNames = StateNames
        self.StateTypes = StateTypes
        self.ExitState = ExitState
        self.StartingPoint = StartingPoint
        # Helps detect errors if other functions are called when Map isn't
        # ready.
        self.mapwidth = -1
        self.mapheight = -1

    def Validate(self):
        """
        Check if Map object has everything it needs.
        """
        Success = True
        Tshape = self.T.shape
        if Tshape[0] != Tshape[2]:
            print("ERROR: Transition matrix has wrong dimensions. MAP-001")
            Success = False
        if Tshape[0] != len(self.S) + 1:  # 1 for the dead state!
            print("ERROR: Transition matrix does not match number of states. MAP-002")
            Success = False
        if Tshape[1] != len(self.A):
            print("ERROR: Transition matrix does not match number of actions. MAP-003")
            Success = False
        # Check that location and locationtype match
        if len(self.ObjectLocations) == 0 or len(self.ObjectTypes) == 0:
            print("ERROR: Missing object locations. MAP-004")
            Success = False
        if len(self.ObjectLocations) != len(self.ObjectTypes):
            print(
                "ERROR: List of locations and list of location types are of different length. MAP-005")
            Success = False
        # Check that location types are ordered
        #  from 0 to len(self.ObjectTypes).
        LocTypes = list(set(self.ObjectTypes))
        if range(max(LocTypes) + 1) != LocTypes:
            print("ERROR: Location types are not ordered correctly (They should be ordered from 0 to N, consecutively). Look at your .ini file in the [Objects] section and the \"ObjectTypes\" entry. MAP-018")
            Success = False
        # Check that objectnames match number of objects
        if self.ObjectNames is not None:
            if len(self.ObjectNames) != len(set(self.ObjectTypes)):
                print("ERROR: Object names do not match number of objects. MAP-006")
                Success = False
        # Check that starting point and exit state are in map
        if self.StartingPoint is not None:
            if self.StartingPoint < 0 or self.StartingPoint >= len(self.S):
                print("ERROR: Starting point is not a state number. MAP-007")
                Success = False
        else:
            print("ERROR: Missing starting point. MAP-008")
            Success = False
        if self.ExitState is not None:
            if self.ExitState < 0 or self.ExitState >= len(self.S):
                print("ERROR: Exit state is not a state number. MAP-009")
                Success = False
        else:
            print("ERROR: Missing exit states. MAP-010")
            Success = False
        # Check that there are no object in exit state.
        if self.ExitState in self.ObjectLocations:
            print("ERROR: Cannot have object on exit state. MAP-022")
        # Check that transition matrix makes sense
        if sum([np.all(np.sum(self.T[:, i, :], axis=1) == 1) for i in range(len(self.A))]) != len(self.A):
            print("ERROR: Transition matrix is not well formed. MAP-011")
            Success = False
        return Success

    def BuildGridWorld(self, x, y, diagonal=True):
        """
        Build a simple grid world with a noiseless transition matrix and an unreachable dead.
        Planner objects take advantage of the dead state to build MDPs that converge faster.

        Args:
            x (int): Map's length
            y (int): Map's height
            diagonal (bool): Can the agent travel diagonally?
        """
        self.mapwidth = x
        self.mapheight = y
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
        if self.ObjectNames == []:
            self.ObjectNames = [
                "Object " + str(i) for i in set(self.ObjectTypes)]
        # From, With, To. Add one for the dead state
        self.T = np.zeros((len(self.S) + 1, len(self.A), len(self.S) + 1))
        # First create dead state structure. All actions leave agent in same
        # place.
        self.T[len(self.S), :, len(self.S)] = 1
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
        Insert a square of some type of terrain. This function rewrites old terrains.
        MAPS are numbered from left to right and from top to bottom, with the first state numbered 1 (not 0).

        Args:
            topleftx (int): x coordinate of top left corner of square (counting from left to right)
            toplefty (int): y coordinate of top left corner of square (counting from top to bottom)
            width (int): Square's width
            height (int): Square's height
        """

        if ((topleftx + width - 1) > self.mapwidth) or ((toplefty + height - 1) > self.mapheight):
            print("ERROR: Square doesn't fit in map. MAP-012")
            return None
        TopLeftState = (toplefty - 1) * self.mapwidth + (topleftx) - 1
        for i in range(height):
            initial = TopLeftState + self.mapwidth * i
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
        if (xval <= 0) or (xval > self.mapwidth):
            print("ERROR: x-coordinate out of bounds (Numbering starts at 1). MAP-013")
            return None
        if (yval <= 0) or (yval > self.mapheight):
            print("EROOR: y-coordinate out of bounds (Numbering starts at 1). MAP-014")
        return (yval - 1) * self.mapwidth + xval - 1

    def GetCoordinates(self, State):
        """
        Transform raw state number into coordinates

        Args:
            State (int): State id

        Returns
            Coordinates (list): x and y coordinates ([x,y])
        """
        if (State >= len(self.S)):
            print("ERROR: State out of bound. MAP-015")
            return None
        yval = int(math.floor(State * 1.0 / self.mapwidth)) + 1
        xval = State - self.mapwidth * (yval - 1) + 1
        return [xval, yval]

    def InsertObjects(self, Locations, ObjectTypes, Organic, ObjectNames=None, SurvivalProb=1):
        """
        Add objects to map.

        Args:
            Locations (list): List of state numbers where objects should be placed
            ObjectTypes (list): List of identifiers about object id
            Organic (list)L List identifyng if objects are organic (organic objects have a future discount on the reward as a function of path length)
            ObjectNames (list): List of names for the objects
            SurvivalProb (float): Probability of organic objects surviving.

        Returns:
            None

        Example: Add five objects on first five states. First two and last three objects are of the same kind, respectively.
        >> InsertTargets([0,1,2,3,4],[0,0,1,1,1],["Object A","Object B"],[False,False])
        >> InsertTargets([22,30],[0,1],["Agent","Object"],[True,False], 0.95)
        """
        # Check that location and locationtype match
        if len(Locations) != len(ObjectTypes):
            print(
                "ERROR: List of locations and list of location types are of different length. MAP-016")
            return None
        # Check that object names match number of objects
        if ObjectNames is not None:
            if len(ObjectNames) != len(set(ObjectTypes)):
                print("ERROR: Object names do not match number of objects. MAP-017")
                return None
        # Not useful to validate object types because user might add targets in more than one step.
        # That can be checked later through the validate() method
        self.ObjectLocations = Locations
        self.ObjectTypes = ObjectTypes
        self.Organic = Organic
        self.ObjectNames = ObjectNames
        self.SurvivalProb = SurvivalProb

    def PullObjectStates(self, Coordinates=True):
        """
        Returns a list of states that have an object in them.

        Args:
            Coordinates (bool): Return raw state numbers or coordinates?
        """
        if not Coordinates:
            return self.ObjectLocations
        else:
            return [self.GetCoordinates(item) for item in self.ObjectLocations]

    def AddTerrainNames(self, StateNames):
        """
        Add names to the states depending on the terrain.

        Args:
            StateNames (list): List of strings with state names
        """
        if len(StateNames) != len(set(self.StateTypes)):
            print(
                "ERROR: List of state names does not match number of state types. MAP-018")
            return None
        self.StateNames = StateNames

    def AddExitState(self, ExitState):
        """
        Add exit state to map
        """
        if ExitState < 0 or ExitState >= len(self.S):
            print("ERROR: Exit state is not a state in the map. MAP-019")
            return None
        self.ExitState = ExitState

    def AddStartingPoint(self, StartingPoint):
        """
        Add starting point to map
        """
        if StartingPoint < 0 or StartingPoint >= len(self.S):
            print("ERROR: Starting point is not a state in the map. MAP-020")
            return None
        self.StartingPoint = StartingPoint

    def PrintMap(self, terrain='*'):
        """
        Print map in ascii

        Args:
            terrain (Character): Character to mark terrains.

        >> MyMap.PrintMap()
        """
        if not self.Validate():
            print("WARNING: Map isn't well formed. May fail to print. MAP-021")
        colors = ['\033[94m', '\033[92m', '\033[93m',
                  '\033[91m', '\033[1m', '\033[4m', '\033[95m']
        endcolor = '\033[0m'
        sys.stdout.write("Action space: " + str(self.ActionNames) + "\n")
        sys.stdout.write("Targets: ")
        if self.ObjectLocations != []:
            sys.stdout.write(str(self.PullObjectStates(True)) + "\n")
        else:
            sys.stdout.write("None\n")
        sys.stdout.write("Exit state: ")
        if self.ExitState is not None:
            sys.stdout.write(str(self.GetCoordinates(self.ExitState)) + "\n\n")
        else:
            sys.stdout.write("None\n")
        # Print color keys
        terrains = list(set(self.StateTypes))
        sys.stdout.write("Terrains: ")
        if self.StateNames == []:
            for i in range(len(terrains)):
                sys.stdout.write(
                    colors[i] + "Terrain " + str(i) + endcolor + " ")
        else:
            for i in range(len(terrains)):
                sys.stdout.write(
                    colors[i] + str(self.StateNames[i]) + endcolor + " ")
        sys.stdout.write("\nItems: ")
        if self.ObjectNames == []:
            for i in range(len(self.ObjectTypes)):
                sys.stdout.write(
                    "Object " + str(i) + " ")
                if self.Organic[i]:
                    sys.stdout.write("(Organic) ")
        else:
            for i in self.ObjectTypes:
                sys.stdout.write(
                    str(self.ObjectNames[i]) + " ")
                if self.Organic[i]:
                    sys.stdout.write("(Organic) ")
        sys.stdout.write("\n")
        if sum(self.Organic) > 0:
            sys.stdout.write("Surirval probability: " +
                             str(self.SurvivalProb) + "\n")
        sys.stdout.write("Map labels: Exit state (E), starting point (S)")
        for i in range(len(self.ObjectNames)):
            sys.stdout.write(", " + self.ObjectNames[i] + "(" + str(i) + ")")
        sys.stdout.write("\n\n")
        ObjectStates = self.PullObjectStates(False)  # Get raw object states
        currstate = 0
        begincolor = endcolor
        for i in range(self.mapheight):
            for j in range(self.mapwidth):
                # Check if state being printed has an object
                character = terrain
                if currstate == self.ExitState:
                    character = 'E'
                if currstate == self.StartingPoint:
                    character = 'S'
                if currstate in ObjectStates:
                    index = ObjectStates.index(currstate)
                    character = self.ObjectTypes[index]
                begincolor = colors[self.StateTypes[self.mapwidth * i + j]]
                sys.stdout.write(begincolor + str(character) + endcolor)
                currstate += 1
            sys.stdout.write("\n")
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
                print(property, ': ', value)
        else:
            for (property, value) in vars(self).iteritems():
                print(property)

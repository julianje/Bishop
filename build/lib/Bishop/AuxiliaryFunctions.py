# -*- coding: utf-8 -*-

"""
Supporting functions for Bishop
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import ConfigParser
import os
import pickle
from PosteriorContainer import *
from Observer import *
from Map import *
from Agent import *


def SaveSamples(Container, Name):
    """
    Save object as a pickle file.

    Args:
        Container (PosteriorContainer): PosteriorContainer object
        Name (string): Filename. Function adds ".p" extension if it's not provided
    """
    if Name[-2:] != ".p":
        Name = Name + ".p"
    pickle.dump(Container, open(Name, "wb"))


def LoadSamples(FileName):
    """
    Load samples from a pickle file

    Args:
        FileName (str): filename

    returns:
        Samples
    """
    Samples = pickle.load(open(FileName, "rb"))
    return Samples


def AnalyzeSamples(FileName):
    """
    Print sample summary from a pickle file

    Args:
        FileName (str): filename
    """
    Samples = pickle.load(open(FileName, "rb"))
    Samples.LongSummary()


def LoadObserver(PostCont):
    """
    Load observer object from a PosteriorContainer object.

    Args:
        Postcont (PosteriorContainer): Posterior container object

    Returns:
        Observer object
    """
    if PostCont.MapFile is None:
        print "No map associated with samples. Cannot load observer."
        return None
    else:
        return LoadEnvironment(PostCont.MapFile)


def ShowAvailableMaps():
    """
    Print list of maps in Bishop library.
    """
    for file in os.listdir(os.path.dirname(__file__) + "/Maps/"):
        if file.endswith(".ini"):
            print file[:-4]


def LoadEnvironment(MapName, Silent=False):
    """
    Load a map. If map isn't found in Bishop's library the
    function searches for the map in your working directory.

    Args:
        MapName (str): Name of map to load
        Silent (bool): If false then function doesn't print map.

    Returns:
        Observer object
    """
    Config = ConfigParser.ConfigParser()
    FilePath = os.path.dirname(__file__) + "/Maps/" + MapName + ".ini"
    #########################
    ## Load .ini map first ##
    #########################
    if not os.path.isfile(FilePath):
        print "Map not in library. Checking local directory..."
        FilePath = MapName + ".ini"
        if not os.path.isfile(FilePath):
            print "ERROR: Map not found."
            return None
    Config.read(FilePath)

    # Agent parameter section
    #########################
    if not Config.has_section("AgentParameters"):
        print "ERROR: AgentParameters block missing."
        return None
    if Config.has_option("AgentParameters", "Prior"):
        Prior = Config.get("AgentParameters", "Prior")
    else:
        print "ERROR: No prior specified in AgentParameters. Use Agent.Priors() to see list of priors"
        return None
    if Config.has_option("AgentParameters", "Restrict"):
        Restrict = Config.getboolean("AgentParameters", "Restrict")
    else:
        print "Setting restrict to false (i.e., all terrains are equal)"
        Restrict = False
    if Config.has_option("AgentParameters", "SoftmaxChoice"):
        SoftmaxChoice = Config.getboolean("AgentParameters", "SoftmaxChoice")
    else:
        print "Softmaxing choices"
        SoftmaxChoice = True
    if Config.has_option("AgentParameters", "SoftmaxAction"):
        SoftmaxAction = Config.getboolean("AgentParameters", "SoftmaxAction")
    else:
        print "Softmaxing actions"
        SoftmaxAction = True
    if Config.has_option("AgentParameters", "choiceTau"):
        choiceTau = Config.getfloat("AgentParameters", "choiceTau")
    else:
        print "Setting choice softmax to 0.01"
        choiceTau = 0.01
    if Config.has_option("AgentParameters", "actionTau"):
        actionTau = Config.getfloat("AgentParameters", "actionTau")
    else:
        print "Setting action softmax to 0.01"
        actionTau = 0.01
    if Config.has_option("AgentParameters", "CostParameters"):
        CostParameters = Config.get("AgentParameters", "CostParameters")
        CostParameters = CostParameters.split()
        CostParameters = [float(i) for i in CostParameters]
    else:
        print "ERROR: Missing cost parameters for prior sampling in AgentParameters block."
        return None
    if Config.has_option("AgentParameters", "RewardParameters"):
        RewardParameters = Config.get("AgentParameters", "RewardParameters")
        RewardParameters = [float(i) for i in RewardParameters.split()]
    else:
        print "ERROR: Missing cost parameters for prior sampling in AgentParameters block."
        return None
    if Config.has_option("AgentParameters", "Apathy"):
        Apathy = Config.getfloat("AgentParameters", "Apathy")
    # Map parameter section
    #######################
    if not Config.has_section("MapParameters"):
        print "ERROR: MapParameters block missing."
        return None
    if Config.has_option("MapParameters", "DiagonalTravel"):
        DiagonalTravel = Config.getboolean(
            "MapParameters", "DiagonalTravel")
    else:
        print "Allowing diagonal travel"
        DiagonalTravel = True
    if Config.has_option("MapParameters", "StartingPoint"):
        StartingPoint = Config.getint(
            "MapParameters", "StartingPoint")
    else:
        print "ERROR: Missing starting point in MapParameters block."
        return None
    if Config.has_option("MapParameters", "ExitState"):
        ExitState = Config.getint(
            "MapParameters", "ExitState")
    else:
        print "ERROR: Missing exit state in MapParameters block."
        return None
    if Config.has_option("MapParameters", "MapName"):
        MapName = Config.get(
            "MapParameters", "MapName")
        TerrainPath = os.path.dirname(__file__) + "/Maps/" + MapName
        f = open(TerrainPath, "r")
        MapLoad = True
        StateTypes = []
        StateNames = []
        mapheight = 0
        for line in iter(f):
            if MapLoad:
                states = [int(i) for i in list(line.rstrip())]
                if states == []:
                    MapLoad = False
                else:
                    mapheight += 1
                    StateTypes.extend(states)
            else:
                statename = line.rstrip()
                if statename != "":
                    StateNames.append(statename)
        f.close()
        mapwidth = len(StateTypes) / mapheight
    else:
        print "ERROR: Missing map name"
        return None
    # Load object information
    #########################
    if not Config.has_section("Objects"):
        print "ERROR: Objects block missing."
        return None
    else:
        if Config.has_option("Objects", "ObjectLocations"):
            ObjectLocations = Config.get("Objects", "ObjectLocations")
            ObjectLocations = [int(i) for i in ObjectLocations.split()]
            HasObjects = True
        else:
            print "WARNING: No objects in map (Agent will always go straight home)."
            HasObjects = False
        if HasObjects:
            if Config.has_option("Objects", "ObjectTypes"):
                ObjectTypes = Config.get("Objects", "ObjectTypes")
                ObjectTypes = [int(i) for i in ObjectTypes.split()]
                if len(ObjectTypes) != len(ObjectLocations):
                    print "Error: ObjectLocations and ObjectTypes should have the same length"
                    return None
            else:
                print "WARNING: No information about object types. Setting all to same kind."
                ObjectTypes = [0] * len(ObjectLocations)
            if Config.has_option("Objects", "ObjectNames"):
                ObjectNames = Config.get("Objects", "ObjectNames")
                ObjectNames = [str(i) for i in ObjectNames.split()]
            else:
                ObjectNames = None
        else:
            ObjectTypes = []
            ObjectNames = None
    # Create objects!
    MyMap = Map()
    MyMap.BuildGridWorld(mapwidth, mapheight, DiagonalTravel)
    MyMap.InsertObjects(ObjectLocations, ObjectTypes, ObjectNames)
    MyMap.StateTypes = StateTypes
    MyMap.StateNames = StateNames
    MyMap.AddStartingPoint(StartingPoint)
    MyMap.AddExitState(ExitState)
    if not Silent:
        MyMap.PrintMap()
    MyAgent = Agent(MyMap, Prior, CostParameters, RewardParameters, SoftmaxChoice, SoftmaxAction, choiceTau, actionTau, Apathy, Restrict)
    return Observer(MyAgent, MyMap)

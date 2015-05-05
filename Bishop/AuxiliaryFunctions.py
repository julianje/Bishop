# -*- coding: utf-8 -*-

"""
Markov Decision Process solver.
"""

__license__ = "MIT"

import ConfigParser
import os
import pickle
from PosteriorContainer import *
from Observer import *
from Map import *
from Agent import *


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

    returns:
        None
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
        print "No map associated with samples."
        return None
    else:
        return LoadEnvironment(PostCont.MapFile)


def ShowAvailableMaps():
    """
    Print list of maps in Bishop library

    Args:
        None
    """
    for file in os.listdir(os.path.dirname(__file__) + "/Maps/"):
        if file.endswith(".ini"):
            print file[:-4]


def LoadEnvironment(MapName):
    """
    Load a map. If map isn't found in Bishop's library it searches for the map in your working directory.

    Args:
        MapName (str): Name of map to load

    Returns:
        Observer object
    """
    Config = ConfigParser.ConfigParser()
    FilePath = os.path.dirname(__file__) + "/Maps/" + MapName + ".ini"
    if not os.path.isfile(FilePath):
        print "Map not in library. Checking local directory..."
        FilePath = MapName + ".ini"
        if not os.path.isfile(FilePath):
            print "ERROR: Map not found."
            return None
    Config.read(FilePath)
    # Agent parameter section
    if not Config.has_section("AgentParameters"):
        print "ERROR: AgentParameters block missing."
        return None
    if Config.has_option("AgentParameters", "CostParameter"):
        CostParam = Config.getfloat("AgentParameters", "CostParameter")
    else:
        print "ERROR: CostParameter missing in AgentParameter block"
        return None
    if Config.has_option("AgentParameters", "RewardParameter"):
        RewardParam = Config.getfloat("AgentParameters", "RewardParameter")
    else:
        print "ERROR: RewardParameter missing in AgentParameter block"
        return None
    # Inference parameter section
    if not Config.has_section("InferenceParameters"):
        print "No inference parameters specified. Using defaults."
        FutureDiscount = 0.9999
        SoftMaxParam = 0.001
        ConvergenceLimit = 0.0001
    else:
        if Config.has_option("InferenceParameters", "FutureDiscount"):
            FutureDiscount = Config.getfloat(
                "InferenceParameters", "FutureDiscount")
        else:
            print "No future discount found. Setting to 0.9999"
            FutureDiscount = 0.9999
        if Config.has_option("InferenceParameters", "SoftMaxParam"):
            SoftMaxParam = Config.getfloat(
                "InferenceParameters", "SoftMaxParam")
        else:
            print "No softmax value found. Setting to 0.001"
            SoftMaxParam = 0.001
        if Config.has_option("InferenceParameters", "ConvergenceLimit"):
            ConvergenceLimit = Config.getfloat(
                "InferenceParameters", "ConvergenceLimit")
        else:
            print "No value iteration thershold found. Setting to 0.9999"
            ConvergenceLimit = 0.9999
    # Main map parameter section
    if not Config.has_section("MainMapParameters"):
        print "ERROR: MainMapParameters block missing."
        return None
    if Config.has_option("MainMapParameters", "MapWidth"):
        MapWidth = Config.getint("MainMapParameters", "MapWidth")
    else:
        print "ERROR: MapWidth missing in MainMapParameter block"
        return None
    if Config.has_option("MainMapParameters", "MapHeight"):
        MapHeight = Config.getint("MainMapParameters", "MapHeight")
    else:
        print "ERROR: MapHeight missing in MainMapParameter block"
        return None
    if Config.has_option("MainMapParameters", "DiagonalTravel"):
        DiagonalTravel = Config.getboolean(
            "MainMapParameters", "DiagonalTravel")
    else:
        print "No diagonal travel specification found. Setting to true."
        DiagonalTravel = True
    if Config.has_option("MainMapParameters", "TerrainInformation"):
        TerrainInformation = Config.get(
            "MainMapParameters", "TerrainInformation")
        TerrainPath = os.path.dirname(__file__) + "/Maps/" + TerrainInformation
        f = open(TerrainPath, "r")
        MapLoad = True
        StateTypes = []
        StateNames = []
        for line in iter(f):
            if MapLoad:
                states = [int(i) for i in list(line.rstrip())]
                if states == []:
                    MapLoad = False
                else:
                    StateTypes.extend(states)
            else:
                statename = line.rstrip()
                if statename != "":
                    StateNames.append(statename)
        f.close()
    else:
        print "No terrain information. Making terrain uniform"
    # Load object information
    if not Config.has_section("MapObjectInformation"):
        print "ERROR: MapObjectInformation missing"
        return None
    else:
        if Config.has_option("MapObjectInformation", "ObjAX"):
            ObjAX = Config.getint("MapObjectInformation", "ObjAX")
        else:
            print "ERROR: Missing x-axis of object A"
            return None
        if Config.has_option("MapObjectInformation", "ObjBX"):
            ObjBX = Config.getint("MapObjectInformation", "ObjBX")
        else:
            print "ERROR: Missing x-axis of object B"
            return None
        if Config.has_option("MapObjectInformation", "ObjAY"):
            ObjAY = Config.getint("MapObjectInformation", "ObjAY")
        else:
            print "ERROR: Missing y-axis of object A"
            return None
        if Config.has_option("MapObjectInformation", "ObjBY"):
            ObjBY = Config.getint("MapObjectInformation", "ObjBY")
        else:
            print "ERROR: Missing y-axis of object B"
            return None
        if Config.has_option("MapObjectInformation", "ObjAType"):
            ObjAType = Config.getint("MapObjectInformation", "ObjAType")
        else:
            print "No information on first object type. Storing it as first object."
            ObjAType = 0
        if Config.has_option("MapObjectInformation", "ObjBType"):
            ObjBType = Config.getint("MapObjectInformation", "ObjBType")
        else:
            print "No information on second object type. Storing it as first object."
            ObjBType = 0
    # Process object locations
    ObjA_Location = (ObjAY - 1) * MapWidth + ObjAX - 1
    ObjB_Location = (ObjBY - 1) * MapWidth + ObjBX - 1
    Targets = [[], [], [], []]
    Targets[ObjAType].append(ObjA_Location)
    Targets[ObjBType].append(ObjB_Location)

    Terrain = Map()
    Terrain.BuildGridWorld(MapWidth, MapHeight, DiagonalTravel)
    Terrain.AddStateNames(StateNames)
    Terrain.StateTypes = StateTypes
    # Add terrains here along with names
    Terrain.InsertTargets(Targets)
    Protagonist = Agent(Terrain, CostParam, RewardParam)
    RObs = Observer(Terrain, Protagonist)
    RObs.AddMapName(MapName)
    RObs.Plr.UpdateSoftMax(SoftMaxParam)
    RObs.Plr.UpdateConvergence(ConvergenceLimit)
    RObs.Plr.UpdateDiscount(FutureDiscount)
    RObs.M.PrintMap()
    return RObs

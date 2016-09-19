# -*- coding: utf-8 -*-

"""
Supporting functions for Bishop
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"

import ConfigParser
import os
import pickle
import pkg_resources
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
    try:
        if Name[-2:] != ".p":
            Name = Name + ".p"
        pickle.dump(Container, open(Name, "wb"))
    except Exception as error:
        print(error)


def LoadSamples(FileName):
    """
    Load samples from a pickle file

    Args:
        FileName (str): filename

    returns:
        Samples
    """
    try:
        Samples = pickle.load(open(FileName, "rb"))
        return Samples
    except Exception as error:
        print(error)


def AnalyzeSamples(FileName):
    """
    Print sample summary from a pickle file

    Args:
        FileName (str): filename
    """
    try:
        Samples = pickle.load(open(FileName, "rb"))
        Samples.LongSummary()
    except Exception as error:
        print(error)


def LoadObserverFromPC(PostCont):
    """
    Load observer object from a PosteriorContainer object.

    Args:
        Postcont (PosteriorContainer): Posterior container object

    Returns:
        Observer object
    """
    if PostCont.MapFile is None:
        print("No map associated with samples. Cannot load observer.")
        return None
    else:
        try:
            Observer = LoadObserver(PostCont.MapFile, False, True)
            Observer.SetStartingPoint(PostCont.StartingPoint, False)
            Observer.PrintMap()
            return Observer
        except Exception as error:
            print(error)


def GetMapList(CurrDir):
    """
    Get list of map files in a directory.

    Args:
        CurrDir (str): Search directory.

    Returns:
        files (list): List of files.å
    """
    files = []
    for File in os.listdir(CurrDir):
        # Check if it's a file
        if File.endswith(".ini"):
            files.append(File)
    return files


def ShowAvailableMaps(Match=""):
    """
    Print list of maps in Bishop library.

    Args:
        Match (String): Only print maps that contain string.
    """
    # Create an empty dictionary.
    results = {}
    BaseDirectory = os.path.dirname(__file__) + "/Maps/"
    # sys.stdout.write(BaseDirectory)
    Files = GetMapList(BaseDirectory)
    if Files != []:
        results['Bishop main maps'] = Files
    for Item in os.listdir(BaseDirectory):
        TempDir = os.path.join(BaseDirectory, Item)
        if os.path.isdir(TempDir):
            # Add to dictionary
            Files = GetMapList(TempDir)
            if Files != []:
                results[Item] = GetMapList(TempDir)
    # print maps
    for key in results:
        HasPrinted = False
        for i in results[key]:
            if Match in i:
                if not HasPrinted:
                    sys.stdout.write(key + ":\n")
                    HasPrinted = True
                sys.stdout.write("\t" + i + "\n")
    sys.stdout.write("\n")


def LocateFile(CurrDir, filename):
    """
    Search for file recursively.

    Args:
        CurrDir (str): Search directory
        filename (str): filename

    Returns:
        dir (str): Location of the file (if found; None otherwise)
    """
    for Currfile in os.listdir(CurrDir):
        test = os.path.join(CurrDir, Currfile)
        if os.path.isdir(test):
            res = LocateFile(test, filename)
            if res is not None:
                return res
        else:
            # it's a file
            if Currfile == filename:
                return CurrDir


def LoadObserver(MapConfig, Revise=False, Silent=False):
    """
    Load a map. If map isn't found in Bishop's library the
    function searches for the map in your working directory.

    Args:
        MapConfig (str): Name of map to load
        Revise (bool): When true, user manually confirms or overrides parameters.
        Silent (bool): If false then function doesn't print map.

    Returns:
        Observer object
    """
    try:
        Local = False
        if Revise:
            sys.stdout.write(
                "\nPress enter to accept the argument or type in the new value to replace it.\n\n")
        Config = ConfigParser.ConfigParser()
        FilePath = os.path.dirname(__file__) + "/Maps/"
        FilePath = LocateFile(FilePath, MapConfig + ".ini")
        if FilePath is not None:
            FilePath = FilePath + "/" + MapConfig + ".ini"
        #########################
        ## Load .ini map first ##
        #########################
        else:
            print("Map not in library. Checking local directory...")
            FilePath = MapConfig + ".ini"
            Local = True
            if not os.path.isfile(FilePath):
                print("ERROR: Map not found.")
                return None
        Config.read(FilePath)
    except Exception as error:
        print(error)

    # Agent parameter section
    #########################
    if not Config.has_section("AgentParameters"):
        print("ERROR: AgentParameters block missing.")
        return None
    if Config.has_option("AgentParameters", "Method"):
        temp = Config.get("AgentParameters", "Method")
        if temp == 'Linear' or temp == 'Rate':
            Method = temp
        else:
            if temp == 'Discount':
                print("Discount method is now integrated with the linear utility method (2.6+). Use organic markers to mark discounts.")
            else:
                print("ERROR: Unknown utility type. Using a linear utility function.")

            Method = "Linear"
    else:
        print("Using a linear utility function (Add a Method in the AgentParameters block to change to 'Rate' utilities).")
        Method = "Linear"
    if Revise:
        temp = raw_input(
            "Utility type (Rate or Linear. Current=" + str(Method) + "):")
        if temp != '':
            if temp == 'Linear' or temp == 'Rate':
                Method = temp
            else:
                print("Not valid. Setting Method to Linear")
                Method = "Linear"
    if Config.has_option("AgentParameters", "Prior"):
        CostPrior = Config.get("AgentParameters", "Prior")
        RewardPrior = CostPrior
        if Revise:
            temp = raw_input("CostPrior (" + str(CostPrior) + "):")
            if temp != '':
                CostPrior = str(temp)
            temp = raw_input("RewardPrior (" + str(RewardPrior) + "):")
            if temp != '':
                RewardPrior = str(temp)
    else:
        if Config.has_option("AgentParameters", "CostPrior"):
            CostPrior = Config.get("AgentParameters", "CostPrior")
            if Revise:
                temp = raw_input("CostPrior (" + str(CostPrior) + "):")
                if temp != '':
                    CostPrior = str(temp)
        else:
            print(
                "WARNING: No cost prior specified in AgentParameters. Use Agent.Priors() to see list of priors")
            return None
        if Config.has_option("AgentParameters", "RewardPrior"):
            RewardPrior = Config.get("AgentParameters", "RewardPrior")
            if Revise:
                temp = raw_input("RewardPrior (" + str(RewardPrior) + "):")
                if temp != '':
                    RewardPrior = str(temp)
        else:
            print("WARNING: No reward prior specified in AgentParameters. Use Agent.Priors() to see list of priors")
            return None
    if Config.has_option("AgentParameters", "Minimum"):
        Minimum = Config.getint("AgentParameters", "Minimum")
    else:
        Minimum = 0
    if Revise:
        temp = raw_input(
            "Minimum objects to collect (" + str(Minimum) + "):")
        if temp != '':
            Minimum = int(temp)
    if Config.has_option("AgentParameters", "Capacity"):
        Capacity = Config.getint("AgentParameters", "Capacity")
    else:
        Capacity = -1
    if Revise:
        temp = raw_input(
            "Agent capacity (" + str(Capacity) + "; -1 = unlimited):")
        if temp != '':
            Capacity = int(temp)
    if Capacity != -1 and Minimum > Capacity:
        sys.stdout.write(
            "ERROR: Agent's minimum number of elements exceed capacity.")
        return None
    if Config.has_option("AgentParameters", "Restrict"):
        Restrict = Config.getboolean("AgentParameters", "Restrict")
    else:
        print("Setting restrict to false (i.e., uncertainty over which terrain is the easiest)")
        Restrict = False
    if Config.has_option("AgentParameters", "SoftmaxChoice"):
        SoftmaxChoice = Config.getboolean("AgentParameters", "SoftmaxChoice")
    else:
        print("Softmaxing choices")
        SoftmaxChoice = True
    if Revise:
        temp = raw_input("Softmax choices (" + str(SoftmaxChoice) + "):")
        if temp != '':
            if temp == 'True':
                SoftmaxChoice = True
            elif temp == 'False':
                SoftmaxChoice = False
            else:
                sys.stdout.write("Not a valid choice. Ignoring.\n")
    if Config.has_option("AgentParameters", "SoftmaxAction"):
        SoftmaxAction = Config.getboolean("AgentParameters", "SoftmaxAction")
    else:
        print("Softmaxing actions")
        SoftmaxAction = True
    if Revise:
        temp = raw_input("Softmax actions (" + str(SoftmaxAction) + "):")
        if temp != '':
            if temp == 'True':
                SoftmaxAction = True
            elif temp == 'False':
                SoftmaxAction = False
            else:
                sys.stdout.write("Not a valid choice. Ignoring.\n")
    if Config.has_option("AgentParameters", "choiceTau"):
        choiceTau = Config.getfloat("AgentParameters", "choiceTau")
    else:
        if SoftmaxChoice:
            print("Setting choice softmax to 0.01")
            choiceTau = 0.01
        else:
            # Doesn't matter; won't be used.
            choiceTau = 0
    if (Revise and SoftmaxChoice):
        temp = raw_input("Choice tau (" + str(choiceTau) + "):")
        if temp != '':
            choiceTau = float(temp)
    if Config.has_option("AgentParameters", "actionTau"):
        actionTau = Config.getfloat("AgentParameters", "actionTau")
    else:
        if SoftmaxAction:
            print("Setting action softmax to 0.01")
            actionTau = 0.01
        else:
            # Doesn't matter; won't be used.
            actionTau = 0
    if (Revise and SoftmaxChoice):
        temp = raw_input("Action tau (" + str(actionTau) + "):")
        if temp != '':
            actionTau = float(temp)
    if Config.has_option("AgentParameters", "CostParameters"):
        CostParameters = Config.get("AgentParameters", "CostParameters")
        if Revise:
            temp = raw_input("Cost parameters (" + str(CostParameters) + "):")
            if temp != '':
                CostParameters = temp
        CostParameters = CostParameters.split()
        CostParameters = [float(i) for i in CostParameters]
    else:
        print("ERROR: Missing cost parameters for prior sampling in AgentParameters block.")
        return None
    if Config.has_option("AgentParameters", "RewardParameters"):
        RewardParameters = Config.get("AgentParameters", "RewardParameters")
        if Revise:
            temp = raw_input(
                "Reward parameters (" + str(RewardParameters) + "):")
            if temp != '':
                RewardParameters = temp
        RewardParameters = [float(i) for i in RewardParameters.split()]
    else:
        print("ERROR: Missing cost parameters for prior sampling in AgentParameters block.")
        return None
    if Config.has_option("AgentParameters", "PNull"):
        CNull = Config.getfloat("AgentParameters", "PNull")
        RNull = CNull
    else:
        if Config.has_option("AgentParameters", "CNull"):
            CNull = Config.getfloat("AgentParameters", "CNull")
        else:
            print("WARNING: No probability of terrains having null cost. Setting to 0.")
            CNull = 0
        if Config.has_option("AgentParameters", "RNull"):
            RNull = Config.getfloat("AgentParameters", "RNull")
        else:
            print("WARNING: No probability of terrains having null cost. Setting to 0.")
            RNull = 0
    if Revise:
        temp = raw_input("Null cost paramter (" + str(CNull) + "):")
        if temp != '':
            CNull = float(temp)
        temp = raw_input("Null reward paramter (" + str(RNull) + "):")
        if temp != '':
            RNull = float(temp)
    # Map parameter section
    #######################
    if not Config.has_section("MapParameters"):
        print("ERROR: MapParameters block missing.")
        return None
    if Config.has_option("MapParameters", "DiagonalTravel"):
        DiagonalTravel = Config.getboolean(
            "MapParameters", "DiagonalTravel")
    else:
        print("Allowing diagonal travel")
        DiagonalTravel = True
    if Revise:
        temp = raw_input("Diagonal travel (" + str(DiagonalTravel) + "):")
        if temp != '':
            if temp == "True":
                DiagonalTravel = True
            elif temp == "False":
                DiagonalTravel = False
            else:
                sys.stdout.write("Not a valid choice. Ignoring.\n")
    if Config.has_option("MapParameters", "StartingPoint"):
        StartingPoint = Config.getint(
            "MapParameters", "StartingPoint")
    else:
        print("ERROR: Missing starting point in MapParameters block.")
        return None
    if Revise:
        temp = raw_input("Starting point (" + str(StartingPoint) + "):")
        if temp != '':
            StartingPoint = int(temp)
    if Config.has_option("MapParameters", "ExitState"):
        ExitState = Config.getint(
            "MapParameters", "ExitState")
    else:
        print("ERROR: Missing exit state in MapParameters block.")
        return None
    if Revise:
        temp = raw_input("Exit state (" + str(ExitState) + "):")
        if temp != '':
            ExitState = int(temp)
    try:
        if Config.has_option("MapParameters", "MapName"):
            MapName = Config.get(
                "MapParameters", "MapName")
            if not Local:
                TerrainPath = os.path.dirname(__file__) + "/Maps/"
                TerrainPath = os.path.join(
                    LocateFile(TerrainPath, MapName), MapName)
            else:
                TerrainPath = MapName
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
            print("ERROR: Missing map name")
            return None
    except:
        print("ERROR: Cannot load map layout.")
        # raise
        return None
    # Load object information
    #########################
    if not Config.has_section("Objects"):
        print("ERROR: Objects block missing.")
        return None
    else:
        if Config.has_option("Objects", "ObjectLocations"):
            ObjectLocations = Config.get("Objects", "ObjectLocations")
            ObjectLocations = [int(i) for i in ObjectLocations.split()]
            HasObjects = True
        else:
            print("WARNING: No objects in map (Agent will always go straight home).")
            HasObjects = False
        if HasObjects:
            if Config.has_option("Objects", "ObjectTypes"):
                ObjectTypes = Config.get("Objects", "ObjectTypes")
                ObjectTypes = [int(i) for i in ObjectTypes.split()]
                if len(ObjectTypes) != len(ObjectLocations):
                    print(
                        "Error: ObjectLocations and ObjectTypes should have the same length")
                    return None
            else:
                print(
                    "WARNING: No information about object types. Setting all to same kind.")
                ObjectTypes = [0] * len(ObjectLocations)
            if Config.has_option("Objects", "ObjectNames"):
                ObjectNames = Config.get("Objects", "ObjectNames")
                ObjectNames = [str(i) for i in ObjectNames.split()]
            else:
                ObjectNames = None
            if Config.has_option("Objects", "Organic"):
                Organic = Config.get("Objects", "Organic")
                Organic = [bool(i) for i in Organic.split()]
            else:
                print("No organic markers. Treating all objects as dead. Add an Organic line to mark if some object types are agents (add probability of death).")
                Organic = [False] * len(ObjectTypes)
            if Config.has_option("Objects", "SurvivalProb"):
                SurvivalProb = Cofig.getfloat("Objects", "SurvivalProb")
                if sum(Organic) == 0:
                    print("You specified a survival probability, but there are no organic objects. Model will work but maybe you specified the map incorrectly.")
                if Revise:
                    temp = raw_input(
                        "Survival probability (" + str(SurvivalProb) + "):")
                    if temp != '':
                        SurvivalProb = float(temp)
            else:
                if sum(Organic) > 0:
                    if Revise:
                        temp = raw_input(
                            "Survival probability (between 0 and 1):")
                        if temp != '':
                            SurvivalProb = float(temp)
                    else:
                        print("Map has organic objects but survival probability not specified. Setting to 0.95; change this by adding a Survival parameter on the Objects block.")
                        SurvivalProb = 0.95
                else:
                    SurvivalProb = 1  # Just to fit in with Planner constructor.
        else:
            ObjectTypes = []
            ObjectNames = None
    # Create objects!
    try:
        MyMap = Map()
        MyMap.BuildGridWorld(mapwidth, mapheight, DiagonalTravel)
        MyMap.InsertObjects(ObjectLocations, ObjectTypes, Organic, ObjectNames, SurvivalProb)
        MyMap.StateTypes = StateTypes
        MyMap.StateNames = StateNames
        MyMap.AddStartingPoint(StartingPoint)
        MyMap.AddExitState(ExitState)
        if not Silent:
            sys.stdout.write("\n")
            MyMap.PrintMap()
        MyAgent = Agent(MyMap, CostPrior, RewardPrior, CostParameters, RewardParameters, Capacity,
                        Minimum, SoftmaxChoice, SoftmaxAction, choiceTau, actionTau, CNull, RNull, Restrict)
        return Observer(MyAgent, MyMap, Method)
    except Exception as error:
        print(error)


def AboutBishop():
    """
    About.
    """
    sys.stdout.write(
        "    ___      ___      ___      ___      ___      ___   \n")
    sys.stdout.write(
        "   /\\  \\    /\\  \\    /\\  \\    /\\__\\    /\\  \\    /\\  \\  \n")
    sys.stdout.write(
        "  /::\\  \\  _\\:\\  \\  /::\\  \\  /:/__/_  /::\\  \\  /::\\  \\ \n")
    sys.stdout.write(
        " /::\\:\\__\\/\\/::\\__\\/\\:\\:\\__\\/::\\/\\__\\/:/\\:\\__\\/::\\:\\__\\\n")
    sys.stdout.write(
        " \\:\\::/  /\\::/\\/__/\\:\\:\\/__/\\/\\::/  /\\:\\/:/  /\\/\\::/  /\n")
    sys.stdout.write(
        "  \\::/  /  \\:\\__\\   \\::/  /   /:/  /  \\::/  /    \\/__/ \n")
    sys.stdout.write(
        "   \\/__/    \\/__/    \\/__/    \\/__/    \\/__/           \n\n")
    sys.stdout.write(
        "Bishop. V " + str(pkg_resources.get_distribution("Bishop").version) + "\n")
    sys.stdout.write("http://github.com/julianje/Bishop\n")
    sys.stdout.write("Julian Jara-Ettinger. jjara@mit.edu\n\n")
    sys.stdout.write(
        "Results using Bishop 1.0.0: Jara-Ettinger, J., Schulz, L. E., & Tenenbaum J. B. (2015). The naive utility calculus: Joint inferences about the costs and rewards of actions. In Proceedings of the 37th Annual Conference of the Cognitive Science Society.\n\n")

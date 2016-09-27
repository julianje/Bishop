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
import copy
from PosteriorContainer import *
from Observer import *
from Map import *
from Agent import *
from DictionaryNavigation import *


def CompareInferences(ContainerA, ContainerB, decimals=1, csv=0, ContAName=None, ContBName=None):
    """
    This function takes two PosteriorContainer objects,
    finds common variables (terrains or objects), and it
    computes the likelihood that the underlying value is
    the same. This likelihood is computed by binning samples
    so the function requires a level of granularity.

    Args:
        ContainerA (PosteriorContainer)
        ContainerB (PosteriorContainer)
        decimals (float): How many decimals to keep in samples?
        ContAName (string): Container A's name
        ContBName (string): Container B's name
        csv (boolean): Should output be printed in a csv-friendly format?
    """
    # Find which dimensions are shared in common.
    SharedCosts = list(
        set(ContainerA.CostNames).intersection(ContainerB.CostNames))
    SharedRewards = list(
        set(ContainerA.ObjectNames).intersection(ContainerB.ObjectNames))
    # Print which things we're not using.
    if not csv:
        sys.stdout.write("Terrain that cannot be compared: ")
        for Cost in ContainerA.CostNames:
            if Cost not in SharedCosts:
                sys.stdout.write(str(Cost) + " ")
        sys.stdout.write("\n")
        sys.stdout.write("Objects that cannot be compared: ")
        for Object in ContainerA.ObjectNames:
            if Object not in SharedRewards:
                sys.stdout.write(str(Object) + " ")
        sys.stdout.write("\n")
    # Get aligned sampled indices.
    # Now, the samples from SharedCost[i] can be accessed using
    # ContainerA.CostSamples(ContainerA_CostIndices[i])
    ContainerA_CostIndices = []
    ContainerB_CostIndices = []
    for Cost in SharedCosts:
        ContainerA_CostIndices.append(ContainerA.CostNames.index(Cost))
        ContainerB_CostIndices.append(ContainerB.CostNames.index(Cost))
    ContainerA_RewardIndices = []
    ContainerB_RewardIndices = []
    for Reward in SharedRewards:
        ContainerA_RewardIndices.append(ContainerA.ObjectNames.index(Reward))
        ContainerB_RewardIndices.append(ContainerB.ObjectNames.index(Reward))
    # For each shared cost, get the vectors, round them, and compute the
    # probability
    if csv:
        if ContAName is not None or ContBName is not None:
            sys.stdout.write("EventA,EventB,")
        sys.stdout.write("decimals,")
        for Cost in SharedCosts:
            sys.stdout.write(str(Cost) + ",")
        for Reward in SharedRewards:
            if Reward == SharedRewards[-1]:
                sys.stdout.write(str(Reward) + "\n")
            else:
                sys.stdout.write(str(Reward) + ",")
        if ContAName is not None or ContBName is not None:
            sys.stdout.write(str(ContAName) + "," + str(ContBName) + ",")
        sys.stdout.write(str(decimals) + ",")
    for i in range(len(SharedCosts)):
        Cost = SharedCosts[i]
        ContainerASamples = np.round(ContainerA.CostSamples[
                                     :, ContainerA_CostIndices[i]], decimals)
        ContainerBSamples = np.round(ContainerB.CostSamples[
                                     :, ContainerB_CostIndices[i]], decimals)
        # Create dictionaries where you add the probabilities.
        # Get the domain.
        # But first get rid of the ugliens np.round returns!
        ContainerASamples = ContainerASamples.tolist()
        ContainerASamples = [ContainerASamples[j][0]
                             for j in range(len(ContainerASamples))]
        ContainerBSamples = ContainerBSamples.tolist()
        ContainerBSamples = [ContainerBSamples[j][0]
                             for j in range(len(ContainerBSamples))]
        Domain = list(np.unique(list(np.unique(ContainerASamples)) +
                                list(np.unique(ContainerBSamples))))
        Probabilities_ContainerA = {}
        Probabilities_ContainerB = {}
        for j in range(len(Domain)):
            Probabilities_ContainerA[Domain[j]] = 0
            Probabilities_ContainerB[Domain[j]] = 0
        # Now loop over samples from containers and populate the dictionaries
        # with the likelihoods.
        for index in range(len(ContainerASamples)):
            Probabilities_ContainerA[ContainerASamples[
                index]] += np.exp(ContainerA.LogLikelihoods[index])
        for index in range(len(ContainerBSamples)):
            Probabilities_ContainerB[ContainerBSamples[
                index]] += np.exp(ContainerB.LogLikelihoods[index])
        # Note: Likelihoods are already normalized in the posterior container
        # Compute the probability that they're the same value by iterating over the domain, multiplying the
        # probabilities, and adding them.
        SameProb = 0
        for Sample in Domain:
            SameProb += Probabilities_ContainerA[Sample] * \
                Probabilities_ContainerB[Sample]
        if csv:
            sys.stdout.write(str(SameProb) + ",")
        else:
            sys.stdout.write("Probability that " + Cost +
                             " is the same: " + str(SameProb) + "\n")
    for i in range(len(SharedRewards)):
        Reward = SharedRewards[i]
        ContainerASamples = np.round(ContainerA.RewardSamples[
                                     :, ContainerA_RewardIndices[i]], decimals)
        ContainerBSamples = np.round(ContainerB.RewardSamples[
                                     :, ContainerB_RewardIndices[i]], decimals)
        # Create dictionaries where you add the probabilities.
        # Get the domain.
        # But first get rid of the ugliens np.round returns!
        ContainerASamples = ContainerASamples.tolist()
        ContainerASamples = [ContainerASamples[j][0]
                             for j in range(len(ContainerASamples))]
        ContainerBSamples = ContainerBSamples.tolist()
        ContainerBSamples = [ContainerBSamples[j][0]
                             for j in range(len(ContainerBSamples))]
        Domain = list(np.unique(list(np.unique(ContainerASamples)) +
                                list(np.unique(ContainerBSamples))))
        Probabilities_ContainerA = {}
        Probabilities_ContainerB = {}
        for j in range(len(Domain)):
            Probabilities_ContainerA[Domain[j]] = 0
            Probabilities_ContainerB[Domain[j]] = 0
        # Now loop over samples from containers and populate the dictionaries
        # with the likelihoods.
        for index in range(len(ContainerASamples)):
            Probabilities_ContainerA[ContainerASamples[
                index]] += np.exp(ContainerA.LogLikelihoods[index])
        for index in range(len(ContainerBSamples)):
            Probabilities_ContainerB[ContainerBSamples[
                index]] += np.exp(ContainerB.LogLikelihoods[index])
        SameProb = 0
        for Sample in Domain:
            SameProb += Probabilities_ContainerA[Sample] * \
                Probabilities_ContainerB[Sample]
        if csv:
            if i == (len(SharedRewards) - 1):
                sys.stdout.write(str(SameProb) + "\n")
            else:
                sys.stdout.write(str(SameProb) + ",")
        else:
            sys.stdout.write("Probability that " + Reward +
                             " is the same: " + str(SameProb) + "\n")


def ProbabilityOfNoChange(ContA, ContB, TestVariable, Conditioning, decimals=1):
    """
    Compute the probability that a random variable has the same value in two events,
    conditioned on one or more random variables being the same. For instance, you can compute
    the probability that a terrain's cost is different, conditioned on the rewards being the same.

    Args:
        ContA (PosteriorContainer): PosteriorContainer object
        ContB (PosteriorContainer): PosteriorContainer object
        TestVariable (string): Random variable to test. Must exist in both containers.
        Conditioning (list of strings): Random variable names to fix across events. Must exist in both containers.
        decimals (int): Decimals to cut off from samples.
    """
    ContainerA = copy.deepcopy(ContA)
    ContainerB = copy.deepcopy(ContB)
    if (ContainerA.CostNames != ContainerB.CostNames) or (ContainerA.ObjectNames != ContainerB.ObjectNames):
        sys.stdout.write(
            "Error: ProbabilityOfChange only supports containers with matched objects and terrains.")
        return None
    # First check that test variable and conditioning exist in both Containers.
    Test = [(x in (ContainerA.CostNames + ContainerA.ObjectNames))
            for x in ([TestVariable] + Conditioning)]
    if sum(Test) != len(Test):
        sys.stdout.write(
            "Error: TestVariable and/or Conditioning variables do not exist on both PosteriorContainer objects. AUXILIARYFUNCTIONS-001")
        return None
    # Round all the samples!
    ContainerA.CostSamples = np.round(ContainerA.CostSamples, decimals)
    ContainerA.RewardSamples = np.round(ContainerA.RewardSamples, decimals)
    ContainerB.CostSamples = np.round(ContainerB.CostSamples, decimals)
    ContainerB.RewardSamples = np.round(ContainerB.RewardSamples, decimals)
    # Loop through samples and get the probabilities of each conditioning
    # variable matching.
    ConditioningProbabilities = {}
    for IndexA in range(ContainerA.Samples):
        for IndexB in range(ContainerB.Samples):
            # Check if sample pair matches conditioning.
            Hit = True
            for ConditioningVar in Conditioning:
                if ConditioningVar in ContainerA.CostNames:
                    if ContainerA.CostSamples[IndexA, ContainerA.CostNames.index(ConditioningVar)] != ContainerB.CostSamples[IndexB, ContainerB.CostNames.index(ConditioningVar)]:
                        Hit = False
                        break
                else:
                    if ContainerA.RewardSamples[IndexA, ContainerA.ObjectNames.index(ConditioningVar)] != ContainerB.RewardSamples[IndexB, ContainerB.ObjectNames.index(ConditioningVar)]:
                        Hit = False
                        break
            if Hit:
                # If we found a sample pair where the conditioning is true,
                # then save the sample.
                # Loop over the Conditiniong variables and add the
                # probabilities.
                SampleHit = []
                for ConditioningVar in Conditioning:
                    if ConditioningVar in ContainerA.CostNames:
                        # We could also get the sample from ContainerB since
                        # they're matched.
                        SampleHit.append(ContainerA.CostSamples[
                                         IndexA, ContainerA.CostNames.index(ConditioningVar)])
                    else:
                        SampleHit.append(ContainerA.RewardSamples[
                                         IndexA, ContainerA.ObjectNames.index(ConditioningVar)])
                Prob = np.exp(ContainerA.LogLikelihoods[
                              IndexA]) * np.exp(ContainerB.LogLikelihoods[IndexB])
                # Save if TestVariable matches
                if TestVariable in ContainerA.CostNames:
                    TestMatch = (ContainerA.CostSamples[IndexA, ContainerA.CostNames.index(
                        TestVariable)] == ContainerB.CostSamples[IndexB, ContainerB.CostNames.index(TestVariable)])
                else:
                    TestMatch = (ContainerA.RewardSamples[IndexA, ContainerA.ObjectNames.index(
                        TestVariable)] == ContainerB.RewardSamples[IndexB, ContainerB.ObjectNames.index(TestVariable)])
                # Now save the results.
                # First dictionary entry is whether the test sample matches.
                # Check if dictionary already has entries. Otherwise build a
                # dictionary inside.
                if TestMatch not in ConditioningProbabilities:
                    ConditioningProbabilities[TestMatch] = {}
                # The next N dictionary depths come from the ConditioningVar
                # variables. For each, try to go in and build dictionaries when
                # necessary.
                CurrentDict = ConditioningProbabilities[TestMatch]
                for SampleIndex in range(len(SampleHit) - 1):
                    Sample = SampleHit[SampleIndex]
                    if Sample not in CurrentDict:
                        CurrentDict[Sample] = {}
                    CurrentDict = CurrentDict[Sample]
                # By now, CurrentDict has the final dictionary. Now check if
                # the final sample has a value already.
                if SampleHit[-1] in CurrentDict:
                    CurrentDict[SampleHit[-1]] += Prob
                else:
                    CurrentDict[SampleHit[-1]] = Prob
    # return ConditioningProbabilities
    # Now that you have the dictionary, compute the probabilities.
    # \sum_{x} p(TestVariable match | Conditioning = x)p( Conditioning = x).
    # Every branch in the dictionary is a value of x.
    # But we need the normalizing constant.
    NormalizingConstant = RecursiveDictionaryExtraction(
        ConditioningProbabilities)
    # Normalize the tree
    ConditioningProbabilities = NormalizeDictionary(ConditioningProbabilities, NormalizingConstant)
    # Get indices for all entires:
    SucessDictionaryEntries = BuildKeyList(ConditioningProbabilities[True])
    FailDictionaryEntries = BuildKeyList(ConditioningProbabilities[False])
    # each line is a sample.
    # Iterate over Success samples
    FullProb = 0
    for IndexPath in SucessDictionaryEntries:
        # Compute (p(Conditioning = IndexPath))
        if IndexPath in FailDictionaryEntries:
            p_fail = RetrieveValue(ConditioningProbabilities[False], IndexPath)
        else:
            p_fail = 0
        p_success = RetrieveValue(ConditioningProbabilities[True], IndexPath)
        # Normalize
        p_IndexPath = p_success + p_fail
        if p_IndexPath > 0:
            p_Match = p_success*1.0 / (p_IndexPath)
            FullProb += p_Match * p_IndexPath
    return FullProb


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
    # stdout.write(BaseDirectory)
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
                Organic = [bool(int(i)) for i in Organic.split()]
            else:
                print("No organic markers. Treating all objects as dead. Add an Organic line to mark if some object types are agents (add probability of death).")
                Organic = [False] * len(ObjectTypes)
            if Config.has_option("Objects", "SurvivalProb"):
                SurvivalProb = Config.getfloat("Objects", "SurvivalProb")
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
                    # Just to fit in with Planner constructor.
                    SurvivalProb = 1
        else:
            ObjectTypes = []
            ObjectNames = None
    # Create objects!
    try:
        MyMap = Map()
        MyMap.BuildGridWorld(mapwidth, mapheight, DiagonalTravel)
        MyMap.InsertObjects(ObjectLocations, ObjectTypes,
                            Organic, ObjectNames, SurvivalProb)
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

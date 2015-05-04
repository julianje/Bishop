import MDP
import numpy as np
import math


class Planner(object):

    """
    Planner class handles the MDP and has additional functions to use the MDPs policy.
    Planner contains an Markov Decision Process object (MDP), a softmax parameter (tau), a future discount parameter (gamma), a value iteration convergence parameter (epsilon)
    and the probability that agents may die (deathprob)
    """

    def __init__(self, diagonal=True, MDP=MDP.MDP(), deathprob=0, tau=0.001, gamma=0.9999, epsilon=0.00001):
        """
        Build a Planner.

        All arguments are option. If no arguments are provided the object contains an empty MDP with prespecified parameters.

        ARGUMENTS:
        diagonal[boolean] determines if agent can travel diagonally
        MDP         MDP object
        deathprob   0<=deathprob<1 is the probability of targets disappearing (Using the last two positions or targets in Map class).
        tau         softmax parameter. Set by default to 0.001
        gamma       future discount parameter. Set by default to 0.9999
        epsilon     value iteration convergence limit. Set by default to 0.00001
        """
        self.MDP = MDP
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.diagonal = diagonal
        # Probability that a target will disappear.
        self.deathprob = deathprob

    def Integrate(self, Agent, Map, ExitType="OneLocation", exitstate=0):
        """
        Integrate builds the MDP using a Map and an Agent object.
        This function calls Planner methods that help build the MDP.
        It doesn't return anything. Instead. the object's MDP will change.
        ARGUMENTS:
        Agent     Agent object
        Map       Map or Map object
        """
        found = 0
        if ExitType == "OneLocation":
            found = 1
            # Agent can only exit from the map in exitstate
            self.MDP = MDP.MDP(range(self.GetDeepStateSize(Map)), Map.A, self.BuildDeepT(
                Agent, Map, [exitstate]), self.BuildDeepR(Agent, Map), self.diagonal, self.gamma)
        if ExitType == "Border":
            found = 1
            # Exit from any map border:
            self.MDP = MDP.MDP(range(self.GetDeepStateSize(Map)), Map.A, self.BuildDeepT(
                Agent, Map), self.BuildDeepR(Agent, Map), self.diagonal, self.gamma)
        if ExitType == "North":
            found = 1
            # Exit from any top location:
            self.MDP = MDP.MDP(range(self.GetDeepStateSize(Map)), Map.A, self.BuildDeepT(
                Agent, Map, False, [range(Map.x)]), self.BuildDeepR(Agent, Map), self.diagonal, self.gamma)
        if found == 0:
            print "ERROR: ExitType does not exit."

    def ComputePolicy(self, Softmax=True):
        """
        ComputePolicy runs value iteration on it's MDP and then builds an optimal policy.
        ARGUMENTS:
        Softmax.   If set to True (Default) the policy will be softmaxed (Using the Planner.tau parameter).
        """
        self.MDP.ValueIteration(self.epsilon)  # Do value iteration on MDP.
        self.MDP.BuildPolicy(self.tau, Softmax)  # Get softmax probabilities.

    def EvalRoute(self, StartingPoint, ActionSequence):
        """
        EvalRoute(StartingPoint,ActionSequence)
        returns the probability of the agent taking the ActionSequence from StartingPoint.
        If there are agents who need help and may die this function is optimistic and assumes no one dies (thus never sending the agent to deathstates)
        """
        # This function retrieves the optimistic state sequence where no
        # objects disappear.
        StateSequence = self.MDP.GetStates(StartingPoint, ActionSequence)
        p = 1  # probability taking the action sequence
        for i in range(len(ActionSequence)):
            p *= (self.MDP.policy[ActionSequence[i], StateSequence[i]])
        return p

    def ExitState(self, State):
        """
        ExitStates(State)
        returns true is State is the Map's death state
        """
        return (State == (self.GetDeepStateSize() - 1))

    def SimulatePathFor(self, StartingPoint, steps):
        """
        SimulatePathFor(StartingPoint,steps)
        simulates an agent's actions for the specified number of steps starting from the starting point
        """
        Actions = [-1] * steps
        State = StartingPoint
        for i in range(steps):
            [State, Actions[i]] = self.MDP.Run(State)
        return Actions

    def SimulatePathUntil(self, StartingPoint, StopStates, Limit, Simple=False):
        """
        Simulate path from StartingPoint until agent reaches a state in the StopStates list.
        Simulation ends after the agent has taken more steps than specified on Limit.
        """
        iterations = 0
        Actions = []
        StateSequence = [StartingPoint]
        State = StartingPoint
        while State not in StopStates:
            [State, NewAct] = self.MDP.Run(State, Simple)
            Actions.append(NewAct)
            StateSequence.append(State)
            iterations += 1
            if (iterations > Limit):
                return [Actions, StateSequence]
        return [Actions, StateSequence]

    def GetDeepStateSize(self, Map):
        """
        GetDeepStateSize(Map) returns the final state size the Planner will use given a Map.
        This function should only be used internally. To check a Map's size use the Map's methods.
        """
        # The map is multiplied into four maps
        # Map 0: Both objects
        # Map 1: Object 1 missing
        # Map 2: Object 2 missing
        # Map 3: No objects
        # Last 2 states are the death state and the exit state.
        return Map.GetWorldSize() * 4 + 2

    def BuildDeepR(self, Agent, Map):
        """
        Builds the MDPs reward functions.
        ARGUMENTS:

        Agent an Agent object.
        Map   a Map of Map object.
        """
        # Note that this function also builds the diagonal rewards,
        # MDP object will discard them if diagonal is off.

        # Build a reward function over the expanded transition matrix.
        # Reward function for actions when you move horizontally or diagonally.
        Rstraight = [0] * self.GetDeepStateSize(Map)
        # Code logic is similar to Planner.BuildRewardFunction.
        # First reward gets inserted in Levels 0 and 2. Second reward gets inserted in Levels 0 and 2.
        # Add costs first
        ActiveStateCosts = [-Agent.costs[Map.StateTypes[i]]
                            for i in range(len(Map.S))] * 4
        ActiveStateCosts.append(0)  # Death state has no cost yet.
        ActiveStateCosts.append(0)  # Exit state has no cost
        Rstraight = ActiveStateCosts
        # Create the diagonal reward matrix
        Rdiagonal = [i * math.sqrt(2) for i in Rstraight]
        # Add rewards
        RewardIndex = 0
        RewardObject = 0  # What object is this reward for?
        for i in range(len(Map.Locations)):
            if Map.Locations[i] != []:
                for j in Map.Locations[i]:
                    Rstraight[j] = Agent.rewards[RewardIndex]
                    Rdiagonal[j] = Agent.rewards[RewardIndex]
                    Rstraight[j + Map.GetWorldSize() * (2 - RewardObject)] = Agent.rewards[
                        RewardIndex]
                    Rdiagonal[j + Map.GetWorldSize() * (2 - RewardObject)] = Agent.rewards[
                        RewardIndex]
                    if i > 1:  # If this an agent-type reward
                        # If Object A is an agent then make the reward for
                        # saving A a cost on all states while A needs help.
                        if (RewardObject == 0):
                            Rstraight[0:Map.GetWorldSize()] = [
                                Rstraight[pos] - Agent.rewards[RewardIndex] for pos in range(Map.GetWorldSize())]
                            Rstraight[(Map.GetWorldSize() * 2):(Map.GetWorldSize() * 3)] = [Rstraight[pos] - Agent.rewards[
                                RewardIndex] for pos in range(Map.GetWorldSize() * 2, Map.GetWorldSize() * 3)]
                            Rdiagonal[0:Map.GetWorldSize()] = [
                                Rdiagonal[pos] - Agent.rewards[RewardIndex] for pos in range(Map.GetWorldSize())]
                            Rdiagonal[(Map.GetWorldSize() * 2):(Map.GetWorldSize() * 3)] = [Rdiagonal[pos] - Agent.rewards[
                                RewardIndex] for pos in range(Map.GetWorldSize() * 2, Map.GetWorldSize() * 3)]
                        else:
                            Rstraight[0:Map.GetWorldSize()] = [
                                Rstraight[pos] - Agent.rewards[RewardIndex] for pos in range(Map.GetWorldSize())]
                            Rstraight[(Map.GetWorldSize()):(Map.GetWorldSize() * 2)] = [Rstraight[pos] - Agent.rewards[
                                RewardIndex] for pos in range(Map.GetWorldSize(), Map.GetWorldSize() * 2)]
                            Rdiagonal[0:Map.GetWorldSize()] = [
                                Rdiagonal[pos] - Agent.rewards[RewardIndex] for pos in range(Map.GetWorldSize())]
                            Rdiagonal[(Map.GetWorldSize()):(Map.GetWorldSize() * 2)] = [Rdiagonal[pos] - Agent.rewards[
                                RewardIndex] for pos in range(Map.GetWorldSize(), Map.GetWorldSize() * 2)]

                    RewardObject += 1
                RewardIndex += 1
        # GIVE A SMALL REWARD FOR GETTING HOME. So MPD can converge in regions
        # where all other rewards were picked up.
        Rstraight[0] = 10
        Rdiagonal[0] = 10
        Rstraight[Map.GetWorldSize()] = 10
        Rdiagonal[Map.GetWorldSize()] = 10
        Rstraight[Map.GetWorldSize() * 2] = 10
        Rdiagonal[Map.GetWorldSize() * 2] = 10
        Rstraight[Map.GetWorldSize() * 3] = 10
        Rdiagonal[Map.GetWorldSize() * 3] = 10
        return [Rstraight, Rdiagonal]

    def BuildDeepT(self, Agent, Map, ExitStates=None):
        """
        Build an deep transition matrix using map information.
        This function takes a Map.T transition matrix and expands it
        so the agent can only pick up each object once.
        ExitStates list is used to build exits.
        """
        if (sum(map(len, Map.Locations)) != 2):
            print "Warning: Didn't find exactly two objects. Don't use Planner.BuildDeepT unless you have exactly two objects."
        WorldSize = Map.GetWorldSize()
        MapExitState = WorldSize * 4 + 1  # Position of exit state.
        # Create big world with death and exit states.
        T = np.zeros(
            (WorldSize * 4 + 2, Map.NumberOfActions(), WorldSize * 4 + 2))
        # Insert the Map's transition matrix into the big matrix.
        T[WorldSize * 0:WorldSize * 1, :, WorldSize * 0:WorldSize * 1] = Map.T
        T[WorldSize * 1:WorldSize * 2, :, WorldSize * 1:WorldSize * 2] = Map.T
        T[WorldSize * 2:WorldSize * 3, :, WorldSize * 2:WorldSize * 3] = Map.T
        T[WorldSize * 3:WorldSize * 4, :, WorldSize * 3:WorldSize * 4] = Map.T
        if (ExitStates == None):
            print "Warning in Planner.BuilDeepT(). No exit states in Map!"
        for exitstate in ExitStates:
            # Check if ExitState is top, left, right, or bottom.
            # Left
            if (exitstate in [loopvar * Map.x for loopvar in range(Map.y)]):
                for DepthLevel in range(4):
                    T[exitstate + WorldSize * DepthLevel, 0,
                        :] = np.zeros((T[exitstate + WorldSize * DepthLevel, 0, :].shape))
                    T[exitstate + WorldSize * DepthLevel, 0, MapExitState] = 1
            # Right
            if (exitstate in [loopvar * Map.x + Map.x - 1 for loopvar in range(Map.y)]):
                for DepthLevel in range(4):
                    T[exitstate + WorldSize * DepthLevel, 1,
                        :] = np.zeros((T[exitstate + WorldSize * DepthLevel, 1, :].shape))
                    T[exitstate + WorldSize * DepthLevel, 1, MapExitState] = 1
            if (exitstate in range(Map.x)):  # Top
                for DepthLevel in range(4):
                    T[exitstate + WorldSize * DepthLevel, 2,
                        :] = np.zeros((T[exitstate + WorldSize * DepthLevel, 2, :].shape))
                    T[exitstate + WorldSize * DepthLevel, 2, MapExitState] = 1
            if (exitstate in range(WorldSize - Map.x, WorldSize)):  # Bottom
                for DepthLevel in range(4):
                    T[exitstate + WorldSize * DepthLevel, 3,
                        :] = np.zeros((T[exitstate + WorldSize * DepthLevel, 3, :].shape))
                    T[exitstate + WorldSize * DepthLevel, 3, MapExitState] = 1
            T[exitstate, :, :] = np.zeros((T[exitstate, :, :].shape))
            T[exitstate + WorldSize, :,
                :] = np.zeros((T[exitstate, :, :].shape))
            T[exitstate + WorldSize * 2, :,
                :] = np.zeros((T[exitstate, :, :].shape))
            T[exitstate + WorldSize * 3, :,
                :] = np.zeros((T[exitstate, :, :].shape))
            T[exitstate, :, MapExitState] = [
                1] * len(T[exitstate, :, MapExitState])
            T[exitstate + WorldSize, :, MapExitState] = [1] * \
                len(T[exitstate, :, MapExitState])
            T[exitstate + WorldSize * 2, :, MapExitState] = [1] * \
                len(T[exitstate, :, MapExitState])
            T[exitstate + WorldSize * 3, :, MapExitState] = [1] * \
                len(T[exitstate, :, MapExitState])
        # Add how picking up objects takes you to different levels.
        # Wit this construction, the agent needs to take one action to pick up
        # the object that will not change it's location.
        Target = 1  # Which target are you inserting?
        for i in range(len(Map.Locations)):
            if Map.Locations[i] != []:
                for j in Map.Locations[i]:
                    if Target == 1:
                        # First delete all previous moves
                        T[j, :, :] = np.zeros((T[j, :, :].shape))
                        # Level 0 goes to level 1
                        T[j, :, j + WorldSize] = [1] * \
                            len(T[j, :, j + WorldSize])
                        # And level 2 takes you to level 3
                        T[j + WorldSize * 2, :,
                            :] = np.zeros((T[j + WorldSize * 2, :, :].shape))
                        T[j + WorldSize * 2, :, j + WorldSize * 3] = [1] * \
                            len(T[j + WorldSize * 2, :, j + WorldSize * 3]
                                )  # Level 2 goes to level 3
                        Target += 1
                        # If the target corresponds to an agent who needs help
                        if (i >= 2):
                            # Take all actions on that level and add the chance of the agent dying.
                            # Loop over subarrays and replace values.
                            for subloop1 in range(WorldSize):
                                for subloop2 in range(WorldSize * 4):
                                    for subloop3 in range(Map.NumberOfActions()):
                                        if(T[subloop1, subloop3, subloop2] == 1):
                                            T[subloop1, subloop3, subloop2] = 1 - \
                                                self.deathprob
                                        if(T[subloop1 + WorldSize * 2, subloop3, subloop2] == 1):
                                            T[subloop1 + WorldSize * 2, subloop3,
                                                subloop2] = 1 - self.deathprob
                            # Add chance of moving to death state in all.
                            T[WorldSize * 0:WorldSize * 1, :, WorldSize * 4] = np.ones(
                                (T[WorldSize * 0:WorldSize * 1, :, WorldSize * 4].shape)) * self.deathprob
                            T[WorldSize * 2:WorldSize * 3, :, WorldSize * 4] = np.ones(
                                (T[WorldSize * 2:WorldSize * 3, :, WorldSize * 4].shape)) * self.deathprob
                    else:
                        # First delete all previous moves
                        T[j, :, :] = np.zeros((T[j, :, :].shape))
                        # Level 0 goes to level 2
                        T[j, :, j + WorldSize * 2] = [1] * \
                            len(T[j, :, j + WorldSize * 2])
                        # And level 1 takes you to level 3
                        T[j + WorldSize, :,
                            :] = np.zeros((T[j + WorldSize, :, :].shape))
                        # Level 1 goes to level 3
                        T[j + WorldSize, :, j + WorldSize * 3] = [1] * \
                            len(T[j + WorldSize, :, j + WorldSize * 3])
                        # If the target corresponds to an agent who needs help
                        if (i >= 2):
                            # Take all actions on that level and add the change of the agent dying.
                            # Loop over subarrays and replace values.
                            for subloop1 in range(WorldSize):
                                for subloop2 in range(WorldSize * 4):
                                    for subloop3 in range(Map.NumberOfActions()):
                                        if(T[subloop1, subloop3, subloop2] == 1):
                                            T[subloop1, subloop3, subloop2] = 1 - \
                                                self.deathprob
                                        if(T[subloop1 + WorldSize, subloop3, subloop2] == 1):
                                            T[subloop1 + WorldSize, subloop3,
                                                subloop2] = 1 - self.deathprob
                            # Add chance of moving to death state in all.
                            T[WorldSize * 0:WorldSize * 1, :, WorldSize * 4] = np.ones(
                                (T[WorldSize * 0:WorldSize * 1, :, WorldSize * 4].shape)) * self.deathprob
                            T[WorldSize * 1:WorldSize * 2, :, WorldSize * 4] = np.ones(
                                (T[WorldSize * 1:WorldSize * 2, :, WorldSize * 4].shape)) * self.deathprob
        # Have death state send you to exit.
        T[WorldSize * 4, :, WorldSize * 4 +
            1] = np.ones((T[WorldSize * 4, :, WorldSize * 4 + 1].shape))
        # Make exit state absorbing.
        T[WorldSize * 4 + 1, :, WorldSize * 4 +
            1] = np.ones((T[WorldSize * 4 + 1, :, WorldSize * 4 + 1].shape))
        return T

    def UpdateDeathProb(self, deathprob):
        self.deathprob = deathprob

    def UpdateSoftMax(self, tau):
        self.tau = tau

    def UpdateConvergence(self, epsilon):
        self.epsilon = epsilon

    def UpdateDiscount(self, gamma):
        self.gamma = gamma

    def Display(self, Full=False):
        # Print class properties
        if Full:
            for (property, value) in vars(self).iteritems():
                print property, ': ', value
        else:
            for (property, value) in vars(self).iteritems():
                print property

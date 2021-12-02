# -*- coding: utf-8 -*-

"""
Store results from agent simulations
"""

import sys
import os.path


class AgentSimulation(object):

    def __init__(self, Costs, Rewards, Actions, States, ObjectNames=None, CostNames=None):
        """
        Create an object that stores simulation results.

        Args:
            Costs (list): List of cost samples
            Rewards (list): List of reward samples
            Actions (list): List of action sequences
            States (list): List of state transitions
        """
        self.Costs = Costs
        self.Rewards = Rewards
        self.Actions = Actions
        self.States = States
        self.ObjectNames = ObjectNames
        self.CostNames = CostNames
        self.CostDimensions = len(self.Costs[0])
        if self.Rewards[0] is not None:
            self.RewardDimensions = len(self.Rewards[0])
        else:
            self.RewardDimensions = None
        self.SampleNo = len(self.Costs)

    def PrintActions(self):
        """
        Pretty print action simulations
        """
        for i in range(len(self.Actions)):
            sys.stdout.write(str(self.Actions[i]) + "\n")

    def SaveCSV(self, filename, overwrite=False):
        """
        Export simulation samples as a .csv file

        Args:
            filename (str): Filename
            overwrite (bool): Overwrite file if it exists?
        """
        Header = ""
        if os.path.isfile(filename) and not overwrite:
            print(("ERROR: File exists, type SaveCSV(\"" + filename + "\",True) to overwrite file."))
        else:
            f = open(filename, 'w')
            # Create header
            if self.ObjectNames is not None:
                for i in range(len(self.ObjectNames)):
                    if i == 0:
                        Header = str(self.ObjectNames[i])
                    else:
                        Header = Header + "," + str(self.ObjectNames[i])
            else:
                for i in range(self.RewardDimensions):
                    if i == 0:
                        Header = "Object" + str(i)
                    else:
                        Header = Header + ",Object" + str(i)
            if self.CostNames is not None:
                for i in self.CostNames:
                    Header = Header + "," + str(i)
            else:
                for i in range(self.CostDimensions):
                    Header = Header + ",Terrain" + str(i)
            Header = Header + ",Actions,States\n"
            f.write(Header)
            # Now add the samples
            for i in range(self.SampleNo):
                for j in range(self.RewardDimensions):
                    if j == 0:
                        NewLine = str(self.Rewards[i][j])
                    else:
                        NewLine = NewLine + "," + str(self.Rewards[i][j])
                for j in range(self.CostDimensions):
                    NewLine = NewLine + "," + str(self.Costs[i][j])
                # Print actions
                NewLine = NewLine + ","
                for action in self.Actions[i]:
                    NewLine = NewLine + str(action) + "-"
                # Print states
                NewLine = NewLine + ","
                for state in self.States[i]:
                    NewLine = NewLine + str(state) + "-"
                NewLine = NewLine + "\n"
                f.write(NewLine)
            f.close()

    def Display(self, Full=True):
        """
        Print object attributes.

        .. Warning::

           This function is for internal use only.

        Args:
            Full (bool): When set to False, function only prints attribute names. Otherwise, it also prints its values.

        Returns:
            standard output summary
        """
        if Full:
            for (property, value) in vars(self).items():
                print((property, ': ', value))
        else:
            for (property, value) in vars(self).items():
                print(property)

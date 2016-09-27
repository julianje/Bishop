# -*- coding: utf-8 -*-

"""
Functions for navigating trees represented as embedded dictionaries.
"""

__author__ = "Julian Jara-Ettinger"
__license__ = "MIT"


def BuildKeyList(Dictionary):
    """
    WARNING: This function is for internal use.
    Return a list of lists where each inner list is chain of valid keys which when input access the dictionary tree and retrieve a numerical sample.
    """
    if isinstance(Dictionary, dict):
        Result = []
        for Key in Dictionary.keys():
            List = BuildKeyList(Dictionary[Key])
            List = [[Key] + x for x in List]
            Result = Result + List
        return Result
    else:
        return [[]]


def NormalizeDictionary(Dictionary, Constant):
    """
    WARNING: This function is for internal use.
    Divide all leaf values by a constant.
    """
    if isinstance(Dictionary, dict):
        for key in Dictionary.keys():
            Dictionary[key] = NormalizeDictionary(Dictionary[key], Constant)
        return Dictionary
    else:
        return Dictionary * 1.0 / Constant


def RetrieveValue(Dictionary, IndexPath):
    """
    WARNING: This function is for internal use.
    Enter dictionary recursively using IndexPath and return leaf value.
    """
    if IndexPath == []:
        return Dictionary
    else:
        return RetrieveValue(Dictionary[IndexPath[0]], IndexPath[1:])


def RecursiveDictionaryExtraction(Dictionary):
    """
    WARNING: This function is for internal use.
    This function goes into a tree structures as embedded dictionaries and returns the sum of all the leaves
    """
    if isinstance(Dictionary, dict):
        Values = [RecursiveDictionaryExtraction(
            Dictionary[x]) for x in Dictionary.keys()]
        return sum(Values)
    else:
        return Dictionary

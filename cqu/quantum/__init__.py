"""The quantum_embedding module provides all solutions for quantum classifying included in cqu.

This module provides quantum classifier classes that can be used to 
train and evaluate quantum classifier algorithms. Although this module is used internally
with the Intergrated Model, This module can be used individually to train and
evaluate quantum classifiers.
"""

from .quantum_classifier import QuantumClassifier
from .quantumsvm import QuantumSVM

__all__ = ["QuantumClassifier", "QuantumSVM"]

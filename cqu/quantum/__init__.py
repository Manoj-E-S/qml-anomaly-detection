"""The quantum_embedding module provides all solutions for quantum classifying included in cqu.

This module provides the main 'QuantumClassifier' class that can be used to 
train and evaluate quantum classifiers. Although this module is used internally
with the Intergrated Model, This module can be used individually to train and
evaluate quantum classifiers.
"""

from .quantum_classifier import QuantumClassifier

__all__ = ["QuantumClassifier"]

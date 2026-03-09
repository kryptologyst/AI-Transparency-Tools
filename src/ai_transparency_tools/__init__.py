"""
AI Transparency Tools - Main Package

A comprehensive Explainable AI toolkit for model interpretability and transparency.
"""

__version__ = "1.0.0"
__author__ = "AI Transparency Tools Team"
__email__ = "contact@example.com"

from .explainers import ModelExplainer
from .methods import (
    SHAPExplainer,
    LIMEExplainer,
    IntegratedGradientsExplainer,
    GradCAMExplainer,
    PartialDependenceExplainer,
    CounterfactualExplainer,
)
from .metrics import ExplanationEvaluator
from .data import DataLoader, DatasetMetadata
from .models import ModelFactory
from .utils import set_seed, get_device

__all__ = [
    "ModelExplainer",
    "SHAPExplainer",
    "LIMEExplainer", 
    "IntegratedGradientsExplainer",
    "GradCAMExplainer",
    "PartialDependenceExplainer",
    "CounterfactualExplainer",
    "ExplanationEvaluator",
    "DataLoader",
    "DatasetMetadata",
    "ModelFactory",
    "set_seed",
    "get_device",
]

"""
SHAP (SHapley Additive exPlanations) implementation for model interpretability.

This module provides comprehensive SHAP-based explanation methods including
TreeExplainer, KernelExplainer, and DeepExplainer for different model types.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn


class SHAPExplainer:
    """
    SHAP-based explanation generator for various model types.
    
    This class provides SHAP explanations using different explainers based on
    the model type (tree-based, neural network, or general black-box).
    """
    
    def __init__(
        self,
        model: Union[BaseEstimator, nn.Module],
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain.
            background_data: Background dataset for KernelExplainer (optional).
            feature_names: Names of features (optional).
            device: Device for neural network models (optional).
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.device = device
        
        # Initialize appropriate explainer based on model type
        self.explainer = self._initialize_explainer()
        
    def _initialize_explainer(self) -> Any:
        """
        Initialize the appropriate SHAP explainer based on model type.
        
        Returns:
            Initialized SHAP explainer object.
        """
        if hasattr(self.model, "tree_"):  # Single tree
            return shap.TreeExplainer(self.model)
        elif hasattr(self.model, "estimators_"):  # Tree ensemble
            return shap.TreeExplainer(self.model)
        elif isinstance(self.model, nn.Module):  # Neural network
            if self.background_data is None:
                raise ValueError("Background data required for neural network models")
            return shap.DeepExplainer(self.model, self.background_data)
        else:  # General black-box model
            if self.background_data is None:
                raise ValueError("Background data required for black-box models")
            return shap.KernelExplainer(self.model.predict_proba, self.background_data)
    
    def explain_global(
        self, 
        X: np.ndarray, 
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate global SHAP explanations.
        
        Args:
            X: Input data to explain.
            max_samples: Maximum number of samples to use (for efficiency).
            
        Returns:
            Dictionary containing SHAP values and feature importance.
        """
        if max_samples is not None and len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # Average across classes for global importance
            shap_values_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            feature_importance = np.mean(shap_values_avg, axis=0)
        else:
            feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "feature_names": self.feature_names,
            "explainer_type": type(self.explainer).__name__,
        }
    
    def explain_local(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Generate local SHAP explanations for individual instances.
        
        Args:
            X: Input instances to explain.
            
        Returns:
            Dictionary containing SHAP values for each instance.
        """
        shap_values = self.explainer.shap_values(X)
        
        return {
            "shap_values": shap_values,
            "feature_names": self.feature_names,
            "explainer_type": type(self.explainer).__name__,
        }
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Get feature importance scores based on mean absolute SHAP values.
        
        Args:
            X: Input data to analyze.
            
        Returns:
            Feature importance scores.
        """
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            # Multi-class case: average across classes
            shap_values_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            return np.mean(shap_values_avg, axis=0)
        else:
            return np.mean(np.abs(shap_values), axis=0)
    
    def plot_summary(self, X: np.ndarray, max_display: int = 10) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            X: Input data to explain.
            max_display: Maximum number of features to display.
        """
        shap_values = self.explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, max_display=max_display)
    
    def plot_waterfall(self, X: np.ndarray, instance_idx: int = 0) -> None:
        """
        Plot SHAP waterfall plot for a specific instance.
        
        Args:
            X: Input data.
            instance_idx: Index of instance to explain.
        """
        shap_values = self.explainer.shap_values(X[instance_idx:instance_idx+1])
        
        if isinstance(shap_values, list):
            # For multi-class, use first class
            shap_values = shap_values[0]
            
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=X[instance_idx],
                feature_names=self.feature_names,
            )
        )
    
    def plot_force(self, X: np.ndarray, instance_idx: int = 0) -> None:
        """
        Plot SHAP force plot for a specific instance.
        
        Args:
            X: Input data.
            instance_idx: Index of instance to explain.
        """
        shap_values = self.explainer.shap_values(X[instance_idx:instance_idx+1])
        
        if isinstance(shap_values, list):
            # For multi-class, use first class
            shap_values = shap_values[0]
            
        shap.force_plot(
            self.explainer.expected_value,
            shap_values[0],
            X[instance_idx],
            feature_names=self.feature_names,
        )
    
    def get_explanation_stability(
        self, 
        X: np.ndarray, 
        n_runs: int = 5,
        noise_level: float = 0.01
    ) -> Dict[str, float]:
        """
        Measure explanation stability across multiple runs with small perturbations.
        
        Args:
            X: Input data to test.
            n_runs: Number of runs for stability test.
            noise_level: Standard deviation of noise to add.
            
        Returns:
            Dictionary containing stability metrics.
        """
        explanations = []
        
        for _ in range(n_runs):
            # Add small noise to test stability
            X_noisy = X + np.random.normal(0, noise_level, X.shape)
            shap_values = self.explainer.shap_values(X_noisy)
            
            if isinstance(shap_values, list):
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values = np.abs(shap_values)
                
            explanations.append(np.mean(shap_values, axis=0))
        
        explanations = np.array(explanations)
        
        # Calculate stability metrics
        mean_explanation = np.mean(explanations, axis=0)
        std_explanation = np.std(explanations, axis=0)
        
        # Coefficient of variation
        cv = np.mean(std_explanation / (np.abs(mean_explanation) + 1e-8))
        
        # Rank correlation between runs
        from scipy.stats import spearmanr
        correlations = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                corr, _ = spearmanr(explanations[i], explanations[j])
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        
        return {
            "coefficient_of_variation": cv,
            "average_rank_correlation": avg_correlation,
            "stability_score": avg_correlation - cv,  # Higher is more stable
        }

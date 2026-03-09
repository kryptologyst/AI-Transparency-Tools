"""
Main ModelExplainer class that integrates all XAI methods.

This module provides a unified interface for generating explanations using
multiple XAI methods including SHAP, LIME, Integrated Gradients, and more.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn

from .methods.shap_explainer import SHAPExplainer
from .methods.lime_explainer import LIMEExplainer
from .methods.integrated_gradients import IntegratedGradientsExplainer
from ..data import DatasetMetadata
from ..utils import get_device, set_seed


class ModelExplainer:
    """
    Unified interface for model explanation using multiple XAI methods.
    
    This class provides a comprehensive toolkit for generating explanations
    using various XAI methods with proper evaluation and visualization.
    """
    
    def __init__(
        self,
        model: Union[BaseEstimator, nn.Module],
        X_train: np.ndarray,
        y_train: np.ndarray,
        metadata: Optional[DatasetMetadata] = None,
        feature_names: Optional[List[str]] = None,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initialize ModelExplainer.
        
        Args:
            model: Trained model to explain.
            X_train: Training data used to fit the model.
            y_train: Training labels.
            metadata: Dataset metadata (optional).
            feature_names: Names of features (optional).
            device: Device for neural network models (optional).
            random_state: Random seed for reproducibility.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.metadata = metadata
        self.device = get_device(device)
        self.random_state = random_state
        
        # Set random seed
        set_seed(random_state)
        
        # Get feature names
        if feature_names is None and metadata is not None:
            self.feature_names = metadata.get_feature_names()
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Initialize explainers
        self._initialize_explainers()
        
    def _initialize_explainers(self) -> None:
        """Initialize all available explainers based on model type."""
        self.explainers = {}
        
        # SHAP explainer (works with most models)
        try:
            self.explainers["shap"] = SHAPExplainer(
                self.model,
                background_data=self.X_train[:100] if len(self.X_train) > 100 else self.X_train,
                feature_names=self.feature_names,
                device=self.device,
            )
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
        
        # LIME explainer (works with most models)
        try:
            self.explainers["lime"] = LIMEExplainer(
                self.model,
                training_data=self.X_train,
                feature_names=self.feature_names,
                mode="tabular",
                random_state=self.random_state,
            )
        except Exception as e:
            print(f"Warning: Could not initialize LIME explainer: {e}")
        
        # Integrated Gradients (neural networks only)
        if isinstance(self.model, nn.Module):
            try:
                self.explainers["integrated_gradients"] = IntegratedGradientsExplainer(
                    self.model,
                    baseline_strategy="zeros",
                    device=self.device,
                )
            except Exception as e:
                print(f"Warning: Could not initialize Integrated Gradients explainer: {e}")
    
    def explain_shap(
        self, 
        X: np.ndarray, 
        explanation_type: str = "local",
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations.
        
        Args:
            X: Input data to explain.
            explanation_type: Type of explanation ('local' or 'global').
            max_samples: Maximum number of samples for global explanations.
            
        Returns:
            Dictionary containing SHAP explanation results.
        """
        if "shap" not in self.explainers:
            raise ValueError("SHAP explainer not available")
        
        if explanation_type == "local":
            return self.explainers["shap"].explain_local(X)
        else:
            return self.explainers["shap"].explain_global(X, max_samples=max_samples)
    
    def explain_lime(
        self, 
        X: np.ndarray, 
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Generate LIME explanations.
        
        Args:
            X: Input instances to explain.
            num_features: Number of top features to include.
            num_samples: Number of samples to generate for LIME.
            
        Returns:
            List of LIME explanation dictionaries.
        """
        if "lime" not in self.explainers:
            raise ValueError("LIME explainer not available")
        
        return self.explainers["lime"].explain_multiple_instances(
            X, num_features=num_features, num_samples=num_samples
        )
    
    def explain_integrated_gradients(
        self, 
        X: np.ndarray, 
        targets: Optional[np.ndarray] = None,
        n_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Generate Integrated Gradients explanations.
        
        Args:
            X: Input data to explain.
            targets: Target class indices (optional).
            n_steps: Number of integration steps.
            
        Returns:
            Dictionary containing Integrated Gradients results.
        """
        if "integrated_gradients" not in self.explainers:
            raise ValueError("Integrated Gradients explainer not available")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        targets_tensor = torch.LongTensor(targets).to(self.device) if targets is not None else None
        
        return self.explainers["integrated_gradients"].explain_batch(
            X_tensor, targets_tensor, n_steps=n_steps
        )
    
    def get_feature_importance(
        self, 
        X: np.ndarray, 
        method: str = "shap",
        aggregate_method: str = "mean",
    ) -> Dict[str, Any]:
        """
        Get feature importance using specified method.
        
        Args:
            X: Input data to analyze.
            method: Method to use ('shap', 'lime', 'integrated_gradients').
            aggregate_method: Method to aggregate importance scores.
            
        Returns:
            Dictionary containing feature importance results.
        """
        if method == "shap":
            if "shap" not in self.explainers:
                raise ValueError("SHAP explainer not available")
            importance = self.explainers["shap"].get_feature_importance(X)
            return {
                "feature_importance": importance,
                "method": "shap",
                "feature_names": self.feature_names,
            }
        
        elif method == "lime":
            if "lime" not in self.explainers:
                raise ValueError("LIME explainer not available")
            return self.explainers["lime"].get_feature_importance(
                X, aggregate_method=aggregate_method
            )
        
        elif method == "integrated_gradients":
            if "integrated_gradients" not in self.explainers:
                raise ValueError("Integrated Gradients explainer not available")
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.explainers["integrated_gradients"].get_feature_importance(
                X_tensor, aggregate_method=aggregate_method
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compare_methods(
        self, 
        X: np.ndarray, 
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare explanations from different methods.
        
        Args:
            X: Input data to explain.
            methods: List of methods to compare (optional, uses all available).
            
        Returns:
            Dictionary containing comparison results.
        """
        if methods is None:
            methods = list(self.explainers.keys())
        
        results = {}
        
        for method in methods:
            if method not in self.explainers:
                continue
            
            try:
                if method == "shap":
                    results[method] = self.explain_shap(X, explanation_type="local")
                elif method == "lime":
                    results[method] = self.explain_lime(X)
                elif method == "integrated_gradients":
                    results[method] = self.explain_integrated_gradients(X)
            except Exception as e:
                print(f"Warning: Could not generate {method} explanation: {e}")
                continue
        
        return results
    
    def evaluate_explanations(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate explanation quality using various metrics.
        
        Args:
            X: Input data to evaluate.
            y: True labels.
            methods: List of methods to evaluate (optional).
            
        Returns:
            Dictionary containing evaluation results.
        """
        if methods is None:
            methods = list(self.explainers.keys())
        
        evaluation_results = {}
        
        for method in methods:
            if method not in self.explainers:
                continue
            
            try:
                if method == "shap":
                    # Evaluate SHAP stability
                    stability = self.explainers["shap"].get_explanation_stability(X)
                    evaluation_results[method] = {
                        "stability": stability,
                        "method": "shap",
                    }
                
                elif method == "lime":
                    # Evaluate LIME faithfulness for a few instances
                    sample_size = min(5, len(X))
                    sample_indices = np.random.choice(len(X), sample_size, replace=False)
                    faithfulness_scores = []
                    
                    for idx in sample_indices:
                        explanation = self.explainers["lime"].explain_instance(X[idx])
                        faithfulness = self.explainers["lime"].evaluate_faithfulness(
                            X[idx], explanation
                        )
                        faithfulness_scores.append(faithfulness["faithfulness_score"])
                    
                    evaluation_results[method] = {
                        "faithfulness": {
                            "mean_score": np.mean(faithfulness_scores),
                            "std_score": np.std(faithfulness_scores),
                            "scores": faithfulness_scores,
                        },
                        "method": "lime",
                    }
                
                elif method == "integrated_gradients":
                    # Evaluate sensitivity
                    X_tensor = torch.FloatTensor(X[:5]).to(self.device)  # Test on first 5 instances
                    sensitivity = self.explainers["integrated_gradients"].evaluate_sensitivity(X_tensor)
                    evaluation_results[method] = {
                        "sensitivity": sensitivity,
                        "method": "integrated_gradients",
                    }
            
            except Exception as e:
                print(f"Warning: Could not evaluate {method}: {e}")
                continue
        
        return evaluation_results
    
    def plot_explanations(
        self, 
        X: np.ndarray, 
        method: str = "shap",
        instance_idx: int = 0,
        max_display: int = 10,
    ) -> None:
        """
        Plot explanations using specified method.
        
        Args:
            X: Input data to explain.
            method: Method to use for plotting.
            instance_idx: Index of instance to plot (for local explanations).
            max_display: Maximum number of features to display.
        """
        if method == "shap":
            if "shap" not in self.explainers:
                raise ValueError("SHAP explainer not available")
            self.explainers["shap"].plot_summary(X, max_display=max_display)
        
        elif method == "lime":
            if "lime" not in self.explainers:
                raise ValueError("LIME explainer not available")
            explanation = self.explainers["lime"].explain_instance(X[instance_idx])
            self.explainers["lime"].plot_explanation(explanation)
        
        elif method == "integrated_gradients":
            if "integrated_gradients" not in self.explainers:
                raise ValueError("Integrated Gradients explainer not available")
            X_tensor = torch.FloatTensor(X[instance_idx:instance_idx+1]).to(self.device)
            explanation = self.explainers["integrated_gradients"].explain_instance(X_tensor)
            self.explainers["integrated_gradients"].plot_attributions(
                X_tensor, explanation["attributions"], self.feature_names
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_available_methods(self) -> List[str]:
        """Get list of available explanation methods."""
        return list(self.explainers.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model and explainers."""
        return {
            "model_type": type(self.model).__name__,
            "feature_names": self.feature_names,
            "num_features": len(self.feature_names),
            "training_samples": len(self.X_train),
            "available_methods": self.get_available_methods(),
            "device": str(self.device),
            "random_state": self.random_state,
        }

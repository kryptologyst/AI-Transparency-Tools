"""
LIME (Local Interpretable Model-agnostic Explanations) implementation.

This module provides LIME-based explanations for both tabular and text data,
with support for different interpretable models and feature selection strategies.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from lime import lime_tabular, lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn


class LIMEExplainer:
    """
    LIME-based explanation generator for various model types.
    
    This class provides local explanations using LIME with support for
    tabular data, text data, and different interpretable models.
    """
    
    def __init__(
        self,
        model: Union[BaseEstimator, nn.Module],
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        mode: str = "tabular",
        random_state: int = 42,
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model to explain.
            training_data: Training data used to fit the model.
            feature_names: Names of features (optional).
            mode: Type of data ('tabular' or 'text').
            random_state: Random seed for reproducibility.
        """
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.mode = mode
        self.random_state = random_state
        
        # Initialize appropriate LIME explainer
        self.explainer = self._initialize_explainer()
        
    def _initialize_explainer(self) -> Any:
        """
        Initialize the appropriate LIME explainer based on data mode.
        
        Returns:
            Initialized LIME explainer object.
        """
        if self.mode == "tabular":
            return lime_tabular.LimeTabularExplainer(
                self.training_data,
                feature_names=self.feature_names,
                mode="classification",
                random_state=self.random_state,
            )
        elif self.mode == "text":
            return LimeTextExplainer(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def explain_instance(
        self, 
        instance: Union[np.ndarray, str], 
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> Dict[str, Any]:
        """
        Explain a single instance using LIME.
        
        Args:
            instance: Instance to explain (array for tabular, string for text).
            num_features: Number of top features to include in explanation.
            num_samples: Number of samples to generate for LIME.
            
        Returns:
            Dictionary containing explanation results.
        """
        if self.mode == "tabular":
            explanation = self.explainer.explain_instance(
                instance,
                self._predict_proba_tabular,
                num_features=num_features,
                num_samples=num_samples,
            )
        else:  # text
            explanation = self.explainer.explain_instance(
                instance,
                self._predict_proba_text,
                num_features=num_features,
            )
        
        # Extract explanation data
        explanation_data = explanation.as_list()
        
        return {
            "explanation": explanation_data,
            "explanation_object": explanation,
            "num_features": num_features,
            "mode": self.mode,
        }
    
    def explain_multiple_instances(
        self, 
        instances: Union[np.ndarray, List[str]], 
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Explain multiple instances using LIME.
        
        Args:
            instances: Instances to explain.
            num_features: Number of top features to include in explanation.
            num_samples: Number of samples to generate for LIME.
            
        Returns:
            List of explanation dictionaries.
        """
        explanations = []
        
        for instance in instances:
            explanation = self.explain_instance(
                instance, num_features=num_features, num_samples=num_samples
            )
            explanations.append(explanation)
        
        return explanations
    
    def _predict_proba_tabular(self, instances: np.ndarray) -> np.ndarray:
        """
        Prediction function for tabular data.
        
        Args:
            instances: Input instances.
            
        Returns:
            Prediction probabilities.
        """
        if isinstance(self.model, nn.Module):
            # Neural network model
            with torch.no_grad():
                X_tensor = torch.FloatTensor(instances)
                predictions = self.model(X_tensor)
                if predictions.dim() == 1:
                    # Binary classification
                    probs = torch.sigmoid(predictions).numpy()
                    return np.column_stack([1 - probs, probs])
                else:
                    # Multi-class classification
                    return torch.softmax(predictions, dim=1).numpy()
        else:
            # Scikit-learn model
            return self.model.predict_proba(instances)
    
    def _predict_proba_text(self, texts: List[str]) -> np.ndarray:
        """
        Prediction function for text data.
        
        Args:
            texts: Input texts.
            
        Returns:
            Prediction probabilities.
        """
        # This is a placeholder - in practice, you'd implement text preprocessing
        # and model prediction here based on your specific text model
        raise NotImplementedError("Text prediction not implemented - customize for your text model")
    
    def get_feature_importance(
        self, 
        instances: Union[np.ndarray, List[str]], 
        num_features: int = 10,
        aggregate_method: str = "mean",
    ) -> Dict[str, Any]:
        """
        Get aggregated feature importance across multiple instances.
        
        Args:
            instances: Instances to analyze.
            num_features: Number of top features to consider.
            aggregate_method: Method to aggregate importance ('mean', 'median', 'max').
            
        Returns:
            Dictionary containing aggregated feature importance.
        """
        explanations = self.explain_multiple_instances(instances, num_features=num_features)
        
        # Extract feature importance scores
        feature_scores = {}
        
        for explanation in explanations:
            for feature, score in explanation["explanation"]:
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(abs(score))
        
        # Aggregate scores
        aggregated_scores = {}
        for feature, scores in feature_scores.items():
            if aggregate_method == "mean":
                aggregated_scores[feature] = np.mean(scores)
            elif aggregate_method == "median":
                aggregated_scores[feature] = np.median(scores)
            elif aggregate_method == "max":
                aggregated_scores[feature] = np.max(scores)
            else:
                raise ValueError(f"Unknown aggregate method: {aggregate_method}")
        
        # Sort by importance
        sorted_features = sorted(
            aggregated_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        return {
            "feature_importance": dict(sorted_features),
            "num_instances": len(instances),
            "aggregate_method": aggregate_method,
        }
    
    def evaluate_faithfulness(
        self, 
        instance: Union[np.ndarray, str], 
        explanation: Dict[str, Any],
        num_features: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate faithfulness of LIME explanation using deletion test.
        
        Args:
            instance: Instance to test.
            explanation: LIME explanation result.
            num_features: Number of top features to test.
            
        Returns:
            Dictionary containing faithfulness metrics.
        """
        if self.mode != "tabular":
            raise NotImplementedError("Faithfulness evaluation only implemented for tabular data")
        
        # Get original prediction
        original_pred = self._predict_proba_tabular(instance.reshape(1, -1))[0]
        
        # Get top features from explanation
        top_features = explanation["explanation"][:num_features]
        feature_indices = [self.feature_names.index(feature) for feature, _ in top_features]
        
        # Test deletion: remove top features one by one
        deletion_scores = []
        instance_copy = instance.copy()
        
        for i, feature_idx in enumerate(feature_indices):
            # Set feature to mean value (or zero)
            instance_copy[feature_idx] = np.mean(self.training_data[:, feature_idx])
            
            # Get prediction with feature removed
            modified_pred = self._predict_proba_tabular(instance_copy.reshape(1, -1))[0]
            
            # Calculate prediction change
            pred_change = np.sum(np.abs(original_pred - modified_pred))
            deletion_scores.append(pred_change)
        
        # Calculate faithfulness metrics
        total_change = np.sum(deletion_scores)
        avg_change = np.mean(deletion_scores)
        
        return {
            "total_prediction_change": total_change,
            "average_prediction_change": avg_change,
            "faithfulness_score": total_change,  # Higher change = more faithful
        }
    
    def plot_explanation(self, explanation: Dict[str, Any], figsize: tuple = (8, 6)) -> None:
        """
        Plot LIME explanation.
        
        Args:
            explanation: LIME explanation result.
            figsize: Figure size for the plot.
        """
        import matplotlib.pyplot as plt
        
        explanation_data = explanation["explanation"]
        features, scores = zip(*explanation_data)
        
        plt.figure(figsize=figsize)
        colors = ['red' if score < 0 else 'blue' for score in scores]
        plt.barh(range(len(features)), scores, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance Score')
        plt.title('LIME Explanation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

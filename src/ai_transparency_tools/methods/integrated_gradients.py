"""
Integrated Gradients implementation for neural network interpretability.

This module provides Integrated Gradients explanations for deep learning models,
with support for different baseline strategies and attribution methods.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, GradientShap, Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt


class IntegratedGradientsExplainer:
    """
    Integrated Gradients explanation generator for neural networks.
    
    This class provides gradient-based explanations using Captum's
    Integrated Gradients implementation with various baseline strategies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        baseline_strategy: str = "zeros",
        device: Optional[str] = None,
    ):
        """
        Initialize Integrated Gradients explainer.
        
        Args:
            model: Trained neural network model.
            baseline_strategy: Strategy for baseline ('zeros', 'mean', 'random').
            device: Device for model computation (optional).
        """
        self.model = model
        self.baseline_strategy = baseline_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize explainer
        self.explainer = IntegratedGradients(self.model)
        
    def _get_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate baseline tensor based on strategy.
        
        Args:
            input_tensor: Input tensor to generate baseline for.
            
        Returns:
            Baseline tensor.
        """
        if self.baseline_strategy == "zeros":
            return torch.zeros_like(input_tensor)
        elif self.baseline_strategy == "mean":
            return torch.mean(input_tensor, dim=0, keepdim=True).expand_as(input_tensor)
        elif self.baseline_strategy == "random":
            return torch.randn_like(input_tensor) * 0.1
        else:
            raise ValueError(f"Unknown baseline strategy: {self.baseline_strategy}")
    
    def explain_instance(
        self, 
        input_tensor: torch.Tensor,
        target: Optional[int] = None,
        n_steps: int = 50,
        return_convergence_delta: bool = False,
    ) -> Dict[str, Any]:
        """
        Explain a single instance using Integrated Gradients.
        
        Args:
            input_tensor: Input tensor to explain.
            target: Target class index (optional, uses predicted class if None).
            n_steps: Number of integration steps.
            return_convergence_delta: Whether to return convergence delta.
            
        Returns:
            Dictionary containing attribution results.
        """
        input_tensor = input_tensor.to(self.device)
        
        # Get baseline
        baseline = self._get_baseline(input_tensor)
        
        # Get target if not provided
        if target is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target = torch.argmax(output, dim=1).item()
        
        # Generate attributions
        attributions = self.explainer.attribute(
            input_tensor,
            baselines=baseline,
            target=target,
            n_steps=n_steps,
            return_convergence_delta=return_convergence_delta,
        )
        
        result = {
            "attributions": attributions,
            "target": target,
            "baseline_strategy": self.baseline_strategy,
            "n_steps": n_steps,
        }
        
        if return_convergence_delta:
            result["convergence_delta"] = attributions[1]
            result["attributions"] = attributions[0]
        
        return result
    
    def explain_batch(
        self, 
        input_tensor: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        n_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Explain a batch of instances using Integrated Gradients.
        
        Args:
            input_tensor: Batch of input tensors to explain.
            targets: Target class indices (optional).
            n_steps: Number of integration steps.
            
        Returns:
            Dictionary containing attribution results.
        """
        input_tensor = input_tensor.to(self.device)
        
        # Get baselines
        baselines = self._get_baseline(input_tensor)
        
        # Get targets if not provided
        if targets is None:
            with torch.no_grad():
                outputs = self.model(input_tensor)
                targets = torch.argmax(outputs, dim=1)
        
        # Generate attributions
        attributions = self.explainer.attribute(
            input_tensor,
            baselines=baselines,
            target=targets,
            n_steps=n_steps,
        )
        
        return {
            "attributions": attributions,
            "targets": targets,
            "baseline_strategy": self.baseline_strategy,
            "n_steps": n_steps,
        }
    
    def get_feature_importance(
        self, 
        input_tensor: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        aggregate_method: str = "mean",
    ) -> Dict[str, Any]:
        """
        Get aggregated feature importance across instances.
        
        Args:
            input_tensor: Input tensors to analyze.
            targets: Target class indices (optional).
            aggregate_method: Method to aggregate importance ('mean', 'sum', 'max').
            
        Returns:
            Dictionary containing aggregated feature importance.
        """
        explanation = self.explain_batch(input_tensor, targets)
        attributions = explanation["attributions"]
        
        # Aggregate attributions
        if aggregate_method == "mean":
            importance = torch.mean(torch.abs(attributions), dim=0)
        elif aggregate_method == "sum":
            importance = torch.sum(torch.abs(attributions), dim=0)
        elif aggregate_method == "max":
            importance = torch.max(torch.abs(attributions), dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate_method}")
        
        return {
            "feature_importance": importance.cpu().numpy(),
            "aggregate_method": aggregate_method,
            "num_instances": input_tensor.shape[0],
        }
    
    def evaluate_sensitivity(
        self, 
        input_tensor: torch.Tensor,
        target: Optional[int] = None,
        noise_levels: List[float] = [0.01, 0.05, 0.1],
        n_runs: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate sensitivity of Integrated Gradients to input perturbations.
        
        Args:
            input_tensor: Input tensor to test.
            target: Target class index (optional).
            noise_levels: List of noise levels to test.
            n_runs: Number of runs per noise level.
            
        Returns:
            Dictionary containing sensitivity metrics.
        """
        # Get original attributions
        original_explanation = self.explain_instance(input_tensor, target)
        original_attributions = original_explanation["attributions"]
        
        sensitivity_results = {}
        
        for noise_level in noise_levels:
            correlations = []
            
            for _ in range(n_runs):
                # Add noise to input
                noise = torch.randn_like(input_tensor) * noise_level
                noisy_input = input_tensor + noise
                
                # Get attributions for noisy input
                noisy_explanation = self.explain_instance(noisy_input, target)
                noisy_attributions = noisy_explanation["attributions"]
                
                # Calculate correlation
                corr = torch.corrcoef(torch.stack([
                    original_attributions.flatten(),
                    noisy_attributions.flatten()
                ]))[0, 1].item()
                
                correlations.append(corr)
            
            sensitivity_results[f"noise_{noise_level}"] = {
                "mean_correlation": np.mean(correlations),
                "std_correlation": np.std(correlations),
                "correlations": correlations,
            }
        
        return sensitivity_results
    
    def plot_attributions(
        self, 
        input_tensor: torch.Tensor,
        attributions: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        figsize: tuple = (12, 8),
    ) -> None:
        """
        Plot Integrated Gradients attributions.
        
        Args:
            input_tensor: Original input tensor.
            attributions: Attribution tensor.
            feature_names: Names of features (optional).
            figsize: Figure size.
        """
        # Convert to numpy
        input_np = input_tensor.cpu().numpy()
        attributions_np = attributions.cpu().numpy()
        
        if len(input_np.shape) == 1:
            # Single instance
            input_np = input_np.reshape(1, -1)
            attributions_np = attributions_np.reshape(1, -1)
        
        n_features = input_np.shape[1]
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot input values
        ax1.bar(range(n_features), input_np[0])
        ax1.set_title("Input Values")
        ax1.set_xlabel("Features")
        ax1.set_ylabel("Value")
        ax1.set_xticks(range(n_features))
        ax1.set_xticklabels(feature_names, rotation=45)
        
        # Plot attributions
        colors = ['red' if x < 0 else 'blue' for x in attributions_np[0]]
        ax2.bar(range(n_features), attributions_np[0], color=colors)
        ax2.set_title("Integrated Gradients Attributions")
        ax2.set_xlabel("Features")
        ax2.set_ylabel("Attribution")
        ax2.set_xticks(range(n_features))
        ax2.set_xticklabels(feature_names, rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_other_methods(
        self, 
        input_tensor: torch.Tensor,
        target: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compare Integrated Gradients with other gradient-based methods.
        
        Args:
            input_tensor: Input tensor to explain.
            target: Target class index (optional).
            
        Returns:
            Dictionary containing comparisons with other methods.
        """
        # Integrated Gradients
        ig_explanation = self.explain_instance(input_tensor, target)
        ig_attributions = ig_explanation["attributions"]
        
        # Gradient SHAP
        gradient_shap = GradientShap(self.model)
        gs_attributions = gradient_shap.attribute(
            input_tensor,
            baselines=self._get_baseline(input_tensor),
            target=target,
        )
        
        # Saliency
        saliency = Saliency(self.model)
        saliency_attributions = saliency.attribute(input_tensor, target=target)
        
        # Calculate correlations
        ig_flat = ig_attributions.flatten()
        gs_flat = gs_attributions.flatten()
        sal_flat = saliency_attributions.flatten()
        
        ig_gs_corr = torch.corrcoef(torch.stack([ig_flat, gs_flat]))[0, 1].item()
        ig_sal_corr = torch.corrcoef(torch.stack([ig_flat, sal_flat]))[0, 1].item()
        gs_sal_corr = torch.corrcoef(torch.stack([gs_flat, sal_flat]))[0, 1].item()
        
        return {
            "integrated_gradients": ig_attributions,
            "gradient_shap": gs_attributions,
            "saliency": saliency_attributions,
            "correlations": {
                "ig_vs_gradient_shap": ig_gs_corr,
                "ig_vs_saliency": ig_sal_corr,
                "gradient_shap_vs_saliency": gs_sal_corr,
            },
        }

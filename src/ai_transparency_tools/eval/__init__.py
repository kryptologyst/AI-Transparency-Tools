"""
Comprehensive evaluation framework for XAI methods.

This module provides evaluation metrics for explanation quality including
faithfulness, stability, fidelity, and robustness measures.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, kendalltau
import torch
import torch.nn as nn


class ExplanationEvaluator:
    """
    Comprehensive evaluator for XAI explanation quality.
    
    This class provides various metrics to assess explanation quality including
    faithfulness, stability, fidelity, and robustness measures.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize explanation evaluator.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
    
    def evaluate_faithfulness(
        self,
        model: Union[Any, nn.Module],
        X: np.ndarray,
        explanations: np.ndarray,
        method: str = "deletion",
        top_k: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate faithfulness of explanations using deletion/insertion tests.
        
        Args:
            model: Trained model to evaluate.
            X: Input data.
            explanations: Explanation scores (feature importance).
            method: Method to use ('deletion' or 'insertion').
            top_k: Number of top features to test.
            
        Returns:
            Dictionary containing faithfulness metrics.
        """
        # Get original predictions
        if isinstance(model, nn.Module):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                original_preds = model(X_tensor).cpu().numpy()
        else:
            original_preds = model.predict_proba(X)
        
        # Get top-k features
        top_features = np.argsort(np.abs(explanations), axis=1)[:, -top_k:]
        
        faithfulness_scores = []
        
        for i in range(len(X)):
            instance = X[i].copy()
            original_pred = original_preds[i]
            
            if method == "deletion":
                # Remove top features (set to mean/zero)
                modified_instance = instance.copy()
                modified_instance[top_features[i]] = np.mean(X, axis=0)[top_features[i]]
                
                if isinstance(model, nn.Module):
                    with torch.no_grad():
                        modified_tensor = torch.FloatTensor(modified_instance.reshape(1, -1))
                        modified_pred = model(modified_tensor).cpu().numpy()[0]
                else:
                    modified_pred = model.predict_proba(modified_instance.reshape(1, -1))[0]
                
                # Calculate prediction change
                pred_change = np.sum(np.abs(original_pred - modified_pred))
                faithfulness_scores.append(pred_change)
            
            elif method == "insertion":
                # Keep only top features (set others to mean/zero)
                modified_instance = np.mean(X, axis=0).copy()
                modified_instance[top_features[i]] = instance[top_features[i]]
                
                if isinstance(model, nn.Module):
                    with torch.no_grad():
                        modified_tensor = torch.FloatTensor(modified_instance.reshape(1, -1))
                        modified_pred = model(modified_tensor).cpu().numpy()[0]
                else:
                    modified_pred = model.predict_proba(modified_instance.reshape(1, -1))[0]
                
                # Calculate prediction similarity
                pred_similarity = 1 - np.sum(np.abs(original_pred - modified_pred))
                faithfulness_scores.append(pred_similarity)
        
        return {
            "faithfulness_score": np.mean(faithfulness_scores),
            "faithfulness_std": np.std(faithfulness_scores),
            "method": method,
            "top_k": top_k,
        }
    
    def evaluate_stability(
        self,
        explainer_func: callable,
        X: np.ndarray,
        n_runs: int = 5,
        noise_level: float = 0.01,
    ) -> Dict[str, float]:
        """
        Evaluate stability of explanations across multiple runs.
        
        Args:
            explainer_func: Function that generates explanations.
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
            explanation = explainer_func(X_noisy)
            explanations.append(explanation)
        
        explanations = np.array(explanations)
        
        # Calculate stability metrics
        mean_explanation = np.mean(explanations, axis=0)
        std_explanation = np.std(explanations, axis=0)
        
        # Coefficient of variation
        cv = np.mean(std_explanation / (np.abs(mean_explanation) + 1e-8))
        
        # Rank correlation between runs
        correlations = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                corr, _ = spearmanr(explanations[i].flatten(), explanations[j].flatten())
                correlations.append(corr)
        
        avg_correlation = np.mean(correlations)
        
        return {
            "coefficient_of_variation": cv,
            "average_rank_correlation": avg_correlation,
            "stability_score": avg_correlation - cv,
            "n_runs": n_runs,
            "noise_level": noise_level,
        }
    
    def evaluate_fidelity(
        self,
        black_box_model: Union[Any, nn.Module],
        surrogate_model: Union[Any, nn.Module],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate fidelity of surrogate model to black box model.
        
        Args:
            black_box_model: Original black box model.
            surrogate_model: Surrogate model trained on explanations.
            X: Input data.
            y: True labels.
            
        Returns:
            Dictionary containing fidelity metrics.
        """
        # Get predictions from both models
        if isinstance(black_box_model, nn.Module):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                black_box_preds = black_box_model(X_tensor).cpu().numpy()
        else:
            black_box_preds = black_box_model.predict_proba(X)
        
        if isinstance(surrogate_model, nn.Module):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                surrogate_preds = surrogate_model(X_tensor).cpu().numpy()
        else:
            surrogate_preds = surrogate_model.predict_proba(X)
        
        # Calculate fidelity metrics
        mse = mean_squared_error(black_box_preds, surrogate_preds)
        r2 = r2_score(black_box_preds, surrogate_preds)
        
        # Accuracy comparison
        black_box_acc = np.mean(np.argmax(black_box_preds, axis=1) == y)
        surrogate_acc = np.mean(np.argmax(surrogate_preds, axis=1) == y)
        
        return {
            "mse": mse,
            "r2_score": r2,
            "black_box_accuracy": black_box_acc,
            "surrogate_accuracy": surrogate_acc,
            "accuracy_difference": abs(black_box_acc - surrogate_acc),
            "fidelity_score": r2,  # Higher R² = better fidelity
        }
    
    def evaluate_robustness(
        self,
        model: Union[Any, nn.Module],
        X: np.ndarray,
        explanations: np.ndarray,
        attack_strengths: List[float] = [0.01, 0.05, 0.1],
    ) -> Dict[str, Any]:
        """
        Evaluate robustness of explanations to adversarial perturbations.
        
        Args:
            model: Model to test.
            X: Input data.
            explanations: Original explanations.
            attack_strengths: List of attack strengths to test.
            
        Returns:
            Dictionary containing robustness metrics.
        """
        robustness_results = {}
        
        for strength in attack_strengths:
            # Generate adversarial examples
            X_adv = self._generate_adversarial_examples(X, explanations, strength)
            
            # Get predictions on adversarial examples
            if isinstance(model, nn.Module):
                with torch.no_grad():
                    X_adv_tensor = torch.FloatTensor(X_adv)
                    adv_preds = model(X_adv_tensor).cpu().numpy()
                    X_tensor = torch.FloatTensor(X)
                    orig_preds = model(X_tensor).cpu().numpy()
            else:
                adv_preds = model.predict_proba(X_adv)
                orig_preds = model.predict_proba(X)
            
            # Calculate robustness metrics
            pred_change = np.mean(np.sum(np.abs(orig_preds - adv_preds), axis=1))
            
            # Calculate explanation similarity
            adv_explanations = self._get_adversarial_explanations(model, X_adv)
            explanation_similarity = self._calculate_explanation_similarity(
                explanations, adv_explanations
            )
            
            robustness_results[f"strength_{strength}"] = {
                "prediction_change": pred_change,
                "explanation_similarity": explanation_similarity,
                "robustness_score": explanation_similarity - pred_change,
            }
        
        return robustness_results
    
    def _generate_adversarial_examples(
        self, 
        X: np.ndarray, 
        explanations: np.ndarray, 
        strength: float
    ) -> np.ndarray:
        """
        Generate adversarial examples using explanation-guided perturbations.
        
        Args:
            X: Original input data.
            explanations: Explanation scores.
            strength: Attack strength.
            
        Returns:
            Adversarial examples.
        """
        # Simple gradient-based attack using explanations as gradients
        X_adv = X.copy()
        
        for i in range(len(X)):
            # Perturb features with high explanation scores
            perturbation = strength * np.sign(explanations[i])
            X_adv[i] = X[i] + perturbation
        
        return X_adv
    
    def _get_adversarial_explanations(
        self, 
        model: Union[Any, nn.Module], 
        X_adv: np.ndarray
    ) -> np.ndarray:
        """
        Get explanations for adversarial examples.
        
        Args:
            model: Model to explain.
            X_adv: Adversarial examples.
            
        Returns:
            Explanations for adversarial examples.
        """
        # This is a placeholder - in practice, you'd use your explainer here
        # For now, return random explanations
        return np.random.randn(*X_adv.shape)
    
    def _calculate_explanation_similarity(
        self, 
        explanations1: np.ndarray, 
        explanations2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two sets of explanations.
        
        Args:
            explanations1: First set of explanations.
            explanations2: Second set of explanations.
            
        Returns:
            Average similarity score.
        """
        similarities = []
        
        for i in range(len(explanations1)):
            corr, _ = spearmanr(explanations1[i], explanations2[i])
            similarities.append(corr)
        
        return np.mean(similarities)
    
    def evaluate_completeness(
        self,
        model: Union[Any, nn.Module],
        X: np.ndarray,
        explanations: np.ndarray,
        top_k_list: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """
        Evaluate completeness of explanations using different top-k values.
        
        Args:
            model: Model to evaluate.
            X: Input data.
            explanations: Explanation scores.
            top_k_list: List of top-k values to test.
            
        Returns:
            Dictionary containing completeness metrics.
        """
        completeness_results = {}
        
        for top_k in top_k_list:
            # Get top-k features
            top_features = np.argsort(np.abs(explanations), axis=1)[:, -top_k:]
            
            # Calculate how much of the prediction is explained by top-k features
            explained_scores = []
            
            for i in range(len(X)):
                instance = X[i].copy()
                
                # Set non-top-k features to mean
                modified_instance = np.mean(X, axis=0).copy()
                modified_instance[top_features[i]] = instance[top_features[i]]
                
                # Get predictions
                if isinstance(model, nn.Module):
                    with torch.no_grad():
                        orig_tensor = torch.FloatTensor(instance.reshape(1, -1))
                        mod_tensor = torch.FloatTensor(modified_instance.reshape(1, -1))
                        orig_pred = model(orig_tensor).cpu().numpy()[0]
                        mod_pred = model(mod_tensor).cpu().numpy()[0]
                else:
                    orig_pred = model.predict_proba(instance.reshape(1, -1))[0]
                    mod_pred = model.predict_proba(modified_instance.reshape(1, -1))[0]
                
                # Calculate completeness score
                completeness = np.sum(np.abs(orig_pred - mod_pred))
                explained_scores.append(completeness)
            
            completeness_results[f"top_{top_k}"] = {
                "mean_completeness": np.mean(explained_scores),
                "std_completeness": np.std(explained_scores),
            }
        
        return completeness_results
    
    def generate_evaluation_report(
        self,
        model: Union[Any, nn.Module],
        X: np.ndarray,
        y: np.ndarray,
        explanations: np.ndarray,
        explainer_name: str = "Unknown",
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model: Model to evaluate.
            X: Input data.
            y: True labels.
            explanations: Explanation scores.
            explainer_name: Name of the explainer method.
            
        Returns:
            Dictionary containing comprehensive evaluation results.
        """
        report = {
            "explainer": explainer_name,
            "dataset_info": {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": len(np.unique(y)),
            },
        }
        
        # Faithfulness evaluation
        report["faithfulness"] = self.evaluate_faithfulness(model, X, explanations)
        
        # Completeness evaluation
        report["completeness"] = self.evaluate_completeness(model, X, explanations)
        
        # Robustness evaluation
        report["robustness"] = self.evaluate_robustness(model, X, explanations)
        
        # Overall score (weighted combination)
        faithfulness_score = report["faithfulness"]["faithfulness_score"]
        completeness_score = np.mean([
            report["completeness"][f"top_{k}"]["mean_completeness"] 
            for k in [1, 3, 5]
        ])
        robustness_score = np.mean([
            report["robustness"][f"strength_{s}"]["robustness_score"]
            for s in [0.01, 0.05, 0.1]
        ])
        
        report["overall_score"] = {
            "faithfulness": faithfulness_score,
            "completeness": completeness_score,
            "robustness": robustness_score,
            "weighted_score": 0.4 * faithfulness_score + 0.3 * completeness_score + 0.3 * robustness_score,
        }
        
        return report

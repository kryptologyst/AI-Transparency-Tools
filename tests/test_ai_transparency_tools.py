"""
Test suite for AI Transparency Tools.

This module provides comprehensive tests for all XAI methods and utilities.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import torch
import torch.nn as nn

# Import our XAI tools
from ai_transparency_tools import ModelExplainer, DataLoader, DatasetMetadata
from ai_transparency_tools.eval import ExplanationEvaluator
from ai_transparency_tools.utils import set_seed, get_device, safe_divide


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        val1 = np.random.random()
        
        set_seed(42)
        val2 = np.random.random()
        
        assert val1 == val2, "Random seed not working properly"
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        
        # Test specific device
        cpu_device = get_device("cpu")
        assert cpu_device.type == "cpu"
    
    def test_safe_divide(self):
        """Test safe division function."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=1.0) == 1.0


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_iris_dataset(self):
        """Test Iris dataset loading."""
        data_loader = DataLoader(random_state=42)
        X, y, metadata = data_loader.load_iris_dataset()
        
        assert X.shape == (150, 4)
        assert len(y) == 150
        assert len(metadata.features) == 4
        assert metadata.target["name"] == "species"
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        data_loader = DataLoader(random_state=42)
        X, y, metadata = data_loader.generate_synthetic_dataset(
            n_samples=100, n_features=5, n_classes=2
        )
        
        assert X.shape == (100, 5)
        assert len(y) == 100
        assert len(metadata.features) == 5
        assert len(np.unique(y)) == 2
    
    def test_data_preprocessing(self):
        """Test data preprocessing."""
        data_loader = DataLoader(random_state=42)
        X, y, metadata = data_loader.load_iris_dataset()
        
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.preprocess_data(
            X, y, metadata, scale_features=True
        )
        
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(y_train) + len(y_val) + len(y_test) == len(y)
        assert X_train.shape[1] == X.shape[1]


class TestDatasetMetadata:
    """Test dataset metadata functionality."""
    
    def test_metadata_creation(self):
        """Test metadata creation."""
        features = [
            {"name": "feature1", "type": "numerical", "range": [0, 1], "monotonic": False, "sensitive": False},
            {"name": "feature2", "type": "categorical", "range": [0, 5], "monotonic": True, "sensitive": False}
        ]
        target = {"name": "target", "type": "classification", "classes": ["A", "B"]}
        
        metadata = DatasetMetadata(features, target, description="Test dataset")
        
        assert len(metadata.get_feature_names()) == 2
        assert len(metadata.get_numerical_features()) == 1
        assert len(metadata.get_categorical_features()) == 1
        assert len(metadata.get_monotonic_features()) == 1
    
    def test_metadata_serialization(self):
        """Test metadata serialization."""
        features = [{"name": "feature1", "type": "numerical", "range": [0, 1], "monotonic": False, "sensitive": False}]
        target = {"name": "target", "type": "classification", "classes": ["A", "B"]}
        
        metadata = DatasetMetadata(features, target)
        
        # Test to_dict
        metadata_dict = metadata.to_dict()
        assert "features" in metadata_dict
        assert "target" in metadata_dict
        
        # Test from_dict
        metadata_restored = DatasetMetadata.from_dict(metadata_dict)
        assert metadata_restored.get_feature_names() == metadata.get_feature_names()


class TestModelExplainer:
    """Test ModelExplainer functionality."""
    
    def setup_method(self):
        """Set up test data and model."""
        # Create synthetic data
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        # Create metadata
        features = [
            {"name": f"feature_{i}", "type": "numerical", "range": [float(X[:, i].min()), float(X[:, i].max())], 
             "monotonic": False, "sensitive": False}
            for i in range(X.shape[1])
        ]
        target = {"name": "target", "type": "classification", "classes": ["class_0", "class_1"]}
        self.metadata = DatasetMetadata(features, target)
        
        self.X_train = X
        self.y_train = y
        self.X_test = X[:10]  # Use first 10 samples for testing
    
    def test_explainer_initialization(self):
        """Test explainer initialization."""
        explainer = ModelExplainer(
            self.model, self.X_train, self.y_train, 
            metadata=self.metadata, random_state=42
        )
        
        assert explainer.model == self.model
        assert len(explainer.feature_names) == 5
        assert "shap" in explainer.get_available_methods()
    
    def test_shap_explanations(self):
        """Test SHAP explanation generation."""
        explainer = ModelExplainer(
            self.model, self.X_train, self.y_train, 
            metadata=self.metadata, random_state=42
        )
        
        # Test local explanations
        explanation = explainer.explain_shap(self.X_test, explanation_type="local")
        
        assert "shap_values" in explanation
        assert "feature_names" in explanation
        assert len(explanation["feature_names"]) == 5
    
    def test_lime_explanations(self):
        """Test LIME explanation generation."""
        explainer = ModelExplainer(
            self.model, self.X_train, self.y_train, 
            metadata=self.metadata, random_state=42
        )
        
        # Test LIME explanations
        explanations = explainer.explain_lime(self.X_test[:3], num_features=3)
        
        assert len(explanations) == 3
        assert "explanation" in explanations[0]
        assert len(explanations[0]["explanation"]) <= 3
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        explainer = ModelExplainer(
            self.model, self.X_train, self.y_train, 
            metadata=self.metadata, random_state=42
        )
        
        importance = explainer.get_feature_importance(self.X_test, method="shap")
        
        assert "feature_importance" in importance
        assert len(importance["feature_importance"]) == 5
        assert all(imp >= 0 for imp in importance["feature_importance"])
    
    def test_method_comparison(self):
        """Test method comparison."""
        explainer = ModelExplainer(
            self.model, self.X_train, self.y_train, 
            metadata=self.metadata, random_state=42
        )
        
        comparison = explainer.compare_methods(self.X_test[:3], methods=["shap", "lime"])
        
        assert "shap" in comparison
        assert "lime" in comparison


class TestExplanationEvaluator:
    """Test explanation evaluation functionality."""
    
    def setup_method(self):
        """Set up test data."""
        # Create synthetic data
        X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        self.X_test = X[:10]
        self.y_test = y[:10]
        
        # Create dummy explanations
        self.explanations = np.random.randn(10, 5)
        
        self.evaluator = ExplanationEvaluator(random_state=42)
    
    def test_faithfulness_evaluation(self):
        """Test faithfulness evaluation."""
        faithfulness = self.evaluator.evaluate_faithfulness(
            self.model, self.X_test, self.explanations, method="deletion", top_k=3
        )
        
        assert "faithfulness_score" in faithfulness
        assert "faithfulness_std" in faithfulness
        assert faithfulness["method"] == "deletion"
        assert faithfulness["top_k"] == 3
    
    def test_stability_evaluation(self):
        """Test stability evaluation."""
        def dummy_explainer(X):
            return np.random.randn(len(X), 5)
        
        stability = self.evaluator.evaluate_stability(
            dummy_explainer, self.X_test, n_runs=3, noise_level=0.01
        )
        
        assert "coefficient_of_variation" in stability
        assert "average_rank_correlation" in stability
        assert "stability_score" in stability
        assert stability["n_runs"] == 3
    
    def test_completeness_evaluation(self):
        """Test completeness evaluation."""
        completeness = self.evaluator.evaluate_completeness(
            self.model, self.X_test, self.explanations, top_k_list=[1, 3, 5]
        )
        
        assert "top_1" in completeness
        assert "top_3" in completeness
        assert "top_5" in completeness
        
        for key, value in completeness.items():
            assert "mean_completeness" in value
            assert "std_completeness" in value
    
    def test_evaluation_report(self):
        """Test comprehensive evaluation report."""
        report = self.evaluator.generate_evaluation_report(
            self.model, self.X_test, self.y_test, self.explanations, "test_method"
        )
        
        assert "explainer" in report
        assert "dataset_info" in report
        assert "faithfulness" in report
        assert "completeness" in report
        assert "overall_score" in report
        
        assert report["explainer"] == "test_method"
        assert report["dataset_info"]["n_samples"] == 10
        assert report["dataset_info"]["n_features"] == 5


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Load data
        data_loader = DataLoader(random_state=42)
        X, y, metadata = data_loader.load_iris_dataset()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.preprocess_data(
            X, y, metadata, scale_features=True
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Initialize explainer
        explainer = ModelExplainer(
            model, X_train, y_train, metadata=metadata, random_state=42
        )
        
        # Generate explanations
        shap_explanation = explainer.explain_shap(X_test[:5], explanation_type="local")
        lime_explanations = explainer.explain_lime(X_test[:3], num_features=3)
        
        # Evaluate explanations
        evaluator = ExplanationEvaluator(random_state=42)
        shap_values = shap_explanation["shap_values"]
        if isinstance(shap_values, list):
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        
        evaluation_report = evaluator.generate_evaluation_report(
            model, X_test[:5], y_test[:5], shap_values, "SHAP"
        )
        
        # Verify results
        assert "shap_values" in shap_explanation
        assert len(lime_explanations) == 3
        assert "overall_score" in evaluation_report
        
        print("End-to-end pipeline test completed successfully!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

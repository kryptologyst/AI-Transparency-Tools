"""
Training and evaluation scripts for AI Transparency Tools.

This module provides command-line scripts for training models, generating explanations,
and evaluating explanation quality.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Import our XAI tools
from ai_transparency_tools import ModelExplainer, DataLoader, DatasetMetadata
from ai_transparency_tools.eval import ExplanationEvaluator
from ai_transparency_tools.utils import set_seed, get_device


class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for demonstration purposes."""
    
    def __init__(self, input_size: int, hidden_sizes: list, num_classes: int, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_model(config: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train model based on configuration."""
    model_config = config["model"]
    model_type = model_config["type"]
    
    if model_type == "random_forest":
        rf_config = model_config["random_forest"]
        model = RandomForestClassifier(**rf_config)
        model.fit(X_train, y_train)
        
    elif model_type == "logistic_regression":
        lr_config = model_config["logistic_regression"]
        model = LogisticRegression(**lr_config)
        model.fit(X_train, y_train)
        
    elif model_type == "neural_network":
        nn_config = model_config["neural_network"]
        device = get_device()
        
        # Create model
        num_classes = len(np.unique(y_train))
        model = SimpleNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=nn_config["hidden_sizes"],
            num_classes=num_classes,
            dropout=nn_config["dropout"]
        ).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=nn_config["learning_rate"])
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.LongTensor(y_train).to(device)
        
        # Training loop
        model.train()
        for epoch in range(nn_config["epochs"]):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{nn_config['epochs']}, Loss: {loss.item():.4f}")
        
        model.eval()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def generate_explanations(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata: DatasetMetadata,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate explanations using configured methods."""
    
    # Initialize explainer
    explainer = ModelExplainer(
        model,
        X_test,  # Using test data as background for simplicity
        y_test,
        metadata=metadata,
        random_state=config["dataset"]["random_state"]
    )
    
    explanations = {}
    explanation_config = config["explanations"]
    
    # SHAP explanations
    if explanation_config["shap"]["enabled"]:
        try:
            print("Generating SHAP explanations...")
            shap_explanation = explainer.explain_shap(
                X_test,
                explanation_type="local",
                max_samples=explanation_config["shap"]["max_samples"]
            )
            explanations["shap"] = {
                "shap_values": shap_explanation["shap_values"],
                "feature_names": shap_explanation["feature_names"],
                "explainer_type": shap_explanation["explainer_type"]
            }
            print("SHAP explanations generated successfully.")
        except Exception as e:
            print(f"Error generating SHAP explanations: {e}")
    
    # LIME explanations
    if explanation_config["lime"]["enabled"]:
        try:
            print("Generating LIME explanations...")
            lime_explanations = explainer.explain_lime(
                X_test[:10],  # Limit for efficiency
                num_features=explanation_config["lime"]["num_features"],
                num_samples=explanation_config["lime"]["num_samples"]
            )
            explanations["lime"] = lime_explanations
            print("LIME explanations generated successfully.")
        except Exception as e:
            print(f"Error generating LIME explanations: {e}")
    
    # Integrated Gradients (for neural networks)
    if (explanation_config["integrated_gradients"]["enabled"] and 
        isinstance(model, nn.Module)):
        try:
            print("Generating Integrated Gradients explanations...")
            ig_explanation = explainer.explain_integrated_gradients(
                X_test[:10],  # Limit for efficiency
                n_steps=explanation_config["integrated_gradients"]["n_steps"]
            )
            explanations["integrated_gradients"] = {
                "attributions": ig_explanation["attributions"].cpu().numpy(),
                "targets": ig_explanation["targets"].cpu().numpy(),
                "baseline_strategy": ig_explanation["baseline_strategy"]
            }
            print("Integrated Gradients explanations generated successfully.")
        except Exception as e:
            print(f"Error generating Integrated Gradients explanations: {e}")
    
    return explanations


def evaluate_explanations(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    explanations: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate explanation quality."""
    
    evaluator = ExplanationEvaluator(random_state=config["dataset"]["random_state"])
    evaluation_results = {}
    
    eval_config = config["evaluation"]
    
    # Evaluate each explanation method
    for method_name, explanation_data in explanations.items():
        print(f"Evaluating {method_name} explanations...")
        
        try:
            if method_name == "shap":
                shap_values = explanation_data["shap_values"]
                if isinstance(shap_values, list):
                    shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                
                evaluation_results[method_name] = evaluator.generate_evaluation_report(
                    model, X_test, y_test, shap_values, method_name
                )
            
            elif method_name == "lime":
                # For LIME, we'll evaluate on a subset
                sample_size = min(5, len(explanation_data))
                lime_values = np.zeros((sample_size, X_test.shape[1]))
                
                for i, explanation in enumerate(explanation_data[:sample_size]):
                    for feature, score in explanation["explanation"]:
                        if feature in explanation_data[0]["explanation"][0]:  # Check if feature exists
                            try:
                                feature_idx = int(feature.split("_")[-1]) if "_" in feature else int(feature)
                                lime_values[i, feature_idx] = abs(score)
                            except:
                                pass
                
                evaluation_results[method_name] = evaluator.generate_evaluation_report(
                    model, X_test[:sample_size], y_test[:sample_size], lime_values, method_name
                )
            
            elif method_name == "integrated_gradients":
                ig_values = explanation_data["attributions"]
                evaluation_results[method_name] = evaluator.generate_evaluation_report(
                    model, X_test[:len(ig_values)], y_test[:len(ig_values)], ig_values, method_name
                )
            
            print(f"{method_name} evaluation completed.")
            
        except Exception as e:
            print(f"Error evaluating {method_name}: {e}")
    
    return evaluation_results


def save_results(
    explanations: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    model_performance: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str
) -> None:
    """Save results to files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save explanations
    if config["output"]["save_explanations"]:
        explanations_file = output_path / "explanations.json"
        with open(explanations_file, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            explanations_json = {}
            for method, data in explanations.items():
                explanations_json[method] = {}
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        explanations_json[method][key] = value.tolist()
                    else:
                        explanations_json[method][key] = value
            json.dump(explanations_json, f, indent=2)
        print(f"Explanations saved to {explanations_file}")
    
    # Save evaluation results
    if config["output"]["save_evaluations"]:
        evaluation_file = output_path / "evaluation_results.json"
        with open(evaluation_file, "w") as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        print(f"Evaluation results saved to {evaluation_file}")
    
    # Save model performance
    performance_file = output_path / "model_performance.json"
    with open(performance_file, "w") as f:
        json.dump(model_performance, f, indent=2)
    print(f"Model performance saved to {performance_file}")
    
    # Save configuration
    config_file = output_path / "config.yaml"
    OmegaConf.save(config, config_file)
    print(f"Configuration saved to {config_file}")


def main():
    """Main function for training and evaluation."""
    
    parser = argparse.ArgumentParser(description="AI Transparency Tools - Training and Evaluation")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--dataset", type=str, default="iris",
                       choices=["iris", "wine", "breast_cancer"],
                       help="Dataset to use")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override with command line arguments
    config.dataset.name = args.dataset
    config.dataset.random_state = args.seed
    config.output.output_dir = args.output_dir
    
    # Set random seed
    set_seed(args.seed)
    
    print(f"Starting AI Transparency Tools experiment...")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {config.model.type}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    print("Loading dataset...")
    data_loader = DataLoader(random_state=args.seed)
    
    if args.dataset == "iris":
        X, y, metadata = data_loader.load_iris_dataset()
    elif args.dataset == "wine":
        X, y, metadata = data_loader.load_wine_dataset()
    elif args.dataset == "breast_cancer":
        X, y, metadata = data_loader.load_breast_cancer_dataset()
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.preprocess_data(
        X, y, metadata,
        scale_features=config.dataset.scale_features,
        test_size=config.dataset.test_size,
        val_size=config.dataset.val_size
    )
    
    print(f"Data loaded: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test samples")
    
    # Train model
    print("Training model...")
    model = train_model(config, X_train, y_train)
    
    # Evaluate model performance
    if isinstance(model, nn.Module):
        device = get_device()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_pred = model(X_test_tensor).cpu().numpy()
            y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred = model.predict(X_test)
        y_pred_classes = y_pred
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    model_performance = {
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred_classes, output_dict=True),
        "model_type": config.model.type,
        "dataset": args.dataset
    }
    
    print(f"Model accuracy: {accuracy:.3f}")
    
    # Generate explanations
    print("Generating explanations...")
    explanations = generate_explanations(model, X_test, y_test, metadata, config)
    
    # Evaluate explanations
    print("Evaluating explanation quality...")
    evaluation_results = evaluate_explanations(model, X_test, y_test, explanations, config)
    
    # Save results
    print("Saving results...")
    save_results(explanations, evaluation_results, model_performance, config, args.output_dir)
    
    print("Experiment completed successfully!")
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {config.model.type}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Explanation methods: {list(explanations.keys())}")
    
    if evaluation_results:
        print("\nExplanation Quality Scores:")
        for method, results in evaluation_results.items():
            if "overall_score" in results:
                overall = results["overall_score"]
                print(f"  {method}: {overall['weighted_score']:.3f}")
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

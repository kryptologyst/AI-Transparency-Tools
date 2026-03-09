"""
Data loading and preprocessing utilities.

This module provides data loading, preprocessing, and metadata management
for various datasets used in XAI experiments.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig


class DatasetMetadata:
    """
    Metadata container for datasets used in XAI experiments.
    
    This class stores information about dataset features, target variables,
    sensitive attributes, and other metadata needed for proper XAI analysis.
    """
    
    def __init__(
        self,
        features: List[Dict[str, Any]],
        target: Dict[str, Any],
        sensitive_attributes: Optional[List[str]] = None,
        description: str = "",
    ):
        """
        Initialize dataset metadata.
        
        Args:
            features: List of feature dictionaries containing name, type, range, etc.
            target: Target variable information including name, type, and classes.
            sensitive_attributes: List of sensitive attribute names.
            description: Human-readable description of the dataset.
        """
        self.features = features
        self.target = target
        self.sensitive_attributes = sensitive_attributes or []
        self.description = description
        
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [f["name"] for f in self.features]
    
    def get_numerical_features(self) -> List[str]:
        """Get list of numerical feature names."""
        return [f["name"] for f in self.features if f["type"] == "numerical"]
    
    def get_categorical_features(self) -> List[str]:
        """Get list of categorical feature names."""
        return [f["name"] for f in self.features if f["type"] == "categorical"]
    
    def get_monotonic_features(self) -> List[str]:
        """Get list of monotonic feature names."""
        return [f["name"] for f in self.features if f.get("monotonic", False)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "features": self.features,
            "target": self.target,
            "sensitive_attributes": self.sensitive_attributes,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Create metadata from dictionary."""
        return cls(**data)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save metadata to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DatasetMetadata":
        """Load metadata from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class DataLoader:
    """
    Data loading and preprocessing utility for XAI experiments.
    
    This class provides methods to load various datasets, preprocess them,
    and split them into train/validation/test sets with proper metadata.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize data loader.
        
        Args:
            random_state: Random seed for reproducible data splits.
        """
        self.random_state = random_state
        
    def load_iris_dataset(self) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
        """
        Load the Iris dataset with metadata.
        
        Returns:
            Tuple of (features, target, metadata).
        """
        data = load_iris()
        X = data.data
        y = data.target
        
        metadata = DatasetMetadata(
            features=[
                {
                    "name": "sepal_length",
                    "type": "numerical",
                    "range": [4.3, 7.9],
                    "monotonic": False,
                    "sensitive": False,
                },
                {
                    "name": "sepal_width", 
                    "type": "numerical",
                    "range": [2.0, 4.4],
                    "monotonic": False,
                    "sensitive": False,
                },
                {
                    "name": "petal_length",
                    "type": "numerical", 
                    "range": [1.0, 6.9],
                    "monotonic": True,
                    "sensitive": False,
                },
                {
                    "name": "petal_width",
                    "type": "numerical",
                    "range": [0.1, 2.5],
                    "monotonic": True,
                    "sensitive": False,
                },
            ],
            target={
                "name": "species",
                "type": "classification",
                "classes": ["setosa", "versicolor", "virginica"],
            },
            description="Iris flower classification dataset with 4 numerical features",
        )
        
        return X, y, metadata
    
    def load_wine_dataset(self) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
        """
        Load the Wine dataset with metadata.
        
        Returns:
            Tuple of (features, target, metadata).
        """
        data = load_wine()
        X = data.data
        y = data.target
        
        metadata = DatasetMetadata(
            features=[
                {
                    "name": f"feature_{i}",
                    "type": "numerical",
                    "range": [float(X[:, i].min()), float(X[:, i].max())],
                    "monotonic": False,
                    "sensitive": False,
                }
                for i in range(X.shape[1])
            ],
            target={
                "name": "wine_class",
                "type": "classification", 
                "classes": [f"class_{i}" for i in range(3)],
            },
            description="Wine classification dataset with 13 numerical features",
        )
        
        return X, y, metadata
    
    def load_breast_cancer_dataset(self) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
        """
        Load the Breast Cancer dataset with metadata.
        
        Returns:
            Tuple of (features, target, metadata).
        """
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        metadata = DatasetMetadata(
            features=[
                {
                    "name": data.feature_names[i],
                    "type": "numerical",
                    "range": [float(X[:, i].min()), float(X[:, i].max())],
                    "monotonic": False,
                    "sensitive": False,
                }
                for i in range(X.shape[1])
            ],
            target={
                "name": "diagnosis",
                "type": "classification",
                "classes": ["malignant", "benign"],
            },
            description="Breast cancer diagnosis dataset with 30 numerical features",
        )
        
        return X, y, metadata
    
    def generate_synthetic_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 2,
        n_informative: int = 5,
        noise: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
        """
        Generate synthetic classification dataset.
        
        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features to generate.
            n_classes: Number of classes.
            n_informative: Number of informative features.
            noise: Amount of noise to add.
            
        Returns:
            Tuple of (features, target, metadata).
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_features - n_informative,
            n_classes=n_classes,
            random_state=self.random_state,
            noise=noise,
        )
        
        metadata = DatasetMetadata(
            features=[
                {
                    "name": f"feature_{i}",
                    "type": "numerical",
                    "range": [float(X[:, i].min()), float(X[:, i].max())],
                    "monotonic": i < n_informative,
                    "sensitive": False,
                }
                for i in range(n_features)
            ],
            target={
                "name": "target",
                "type": "classification",
                "classes": [f"class_{i}" for i in range(n_classes)],
            },
            description=f"Synthetic classification dataset with {n_features} features",
        )
        
        return X, y, metadata
    
    def preprocess_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: DatasetMetadata,
        scale_features: bool = True,
        test_size: float = 0.2,
        val_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data and split into train/val/test sets.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            metadata: Dataset metadata.
            scale_features: Whether to scale numerical features.
            test_size: Proportion of data for test set.
            val_size: Proportion of remaining data for validation set.
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_from_file(
        self, 
        filepath: Union[str, Path],
        target_column: str,
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
        """
        Load dataset from CSV file.
        
        Args:
            filepath: Path to CSV file.
            target_column: Name of target column.
            metadata_path: Optional path to metadata JSON file.
            
        Returns:
            Tuple of (features, target, metadata).
        """
        df = pd.read_csv(filepath)
        
        # Separate features and target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        # Load or create metadata
        if metadata_path and Path(metadata_path).exists():
            metadata = DatasetMetadata.load(metadata_path)
        else:
            # Create basic metadata
            feature_names = df.drop(columns=[target_column]).columns.tolist()
            metadata = DatasetMetadata(
                features=[
                    {
                        "name": name,
                        "type": "numerical",  # Assume numerical by default
                        "range": [float(df[name].min()), float(df[name].max())],
                        "monotonic": False,
                        "sensitive": False,
                    }
                    for name in feature_names
                ],
                target={
                    "name": target_column,
                    "type": "classification",  # Assume classification by default
                    "classes": list(np.unique(y)),
                },
                description=f"Dataset loaded from {filepath}",
            )
        
        return X, y, metadata

"""
Utility functions for AI Transparency Tools.

This module provides common utilities including device management, seeding,
and helper functions used throughout the package.
"""

import random
from typing import Optional, Union
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value to use for all random number generators.
        
    Note:
        This function sets seeds for Python's random module, NumPy, and PyTorch.
        For complete reproducibility, call this function before any random operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for PyTorch operations.
    
    Args:
        device: Specific device to use. If None, auto-detect best available device.
                Options: 'cpu', 'cuda', 'mps', or None for auto-detection.
    
    Returns:
        PyTorch device object.
        
    Note:
        Device priority: CUDA > MPS (Apple Silicon) > CPU
        Falls back to CPU if preferred device is not available.
    """
    if device is not None:
        return torch.device(device)
    
    # Auto-detect best available device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        OmegaConf DictConfig object containing the configuration.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: OmegaConf DictConfig object to save.
        config_path: Path where to save the configuration file.
    """
    OmegaConf.save(config, config_path)


def validate_config(config: DictConfig, required_keys: list[str]) -> None:
    """
    Validate that configuration contains required keys.
    
    Args:
        config: Configuration object to validate.
        required_keys: List of required configuration keys.
        
    Raises:
        ValueError: If any required keys are missing from the configuration.
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def format_number(value: Union[int, float], precision: int = 3) -> str:
    """
    Format a number with specified precision.
    
    Args:
        value: Number to format.
        precision: Number of decimal places to show.
        
    Returns:
        Formatted number string.
    """
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}f}"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default value if denominator is zero.
    
    Args:
        numerator: Number to divide.
        denominator: Number to divide by.
        default: Value to return if denominator is zero.
        
    Returns:
        Division result or default value.
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator

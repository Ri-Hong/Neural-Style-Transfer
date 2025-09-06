"""Training utilities including W&B integration."""

import os
from typing import Dict, Any
import wandb
import torch
from omegaconf import DictConfig

def init_wandb(config: DictConfig) -> None:
    """Initialize Weights & Biases logging.
    
    Args:
        config: Hydra configuration object
    """
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        tags=config.wandb.tags,
        config=dict(config),
    )

def setup_training(config: DictConfig) -> Dict[str, Any]:
    """Set up training environment and optimizations.
    
    Args:
        config: Hydra configuration object
        
    Returns:
        Dict containing training setup info
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up optimizations
    if config.optimization.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    
    if config.optimization.channels_last:
        # Enable channels last memory format for better performance on NVIDIA GPUs
        torch.backends.cuda.preferred_memory_format = "channels_last"
    
    return {
        "device": device,
        "amp": config.optimization.amp,
        "compile": config.optimization.compile,
    }

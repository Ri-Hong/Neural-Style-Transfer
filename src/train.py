"""Training script for neural style transfer."""

import hydra
from omegaconf import DictConfig
from utils.train_utils import init_wandb, setup_training

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration object
    """
    # Initialize wandb
    init_wandb(cfg)
    
    # Setup training environment
    train_setup = setup_training(cfg)
    
    print("Configuration:")
    print(f"- Using device: {train_setup['device']}")
    print(f"- AMP enabled: {train_setup['amp']}")
    print(f"- Model compilation: {train_setup['compile']}")
    print(f"- Batch size: {cfg.training.batch_size}")
    print(f"- Learning rate: {cfg.optimizer.lr}")

if __name__ == "__main__":
    train()

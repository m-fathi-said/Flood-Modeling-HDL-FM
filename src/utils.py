import os
import torch
import numpy as np
from config import SEED

def set_random_seeds(seed=SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def create_output_directory(path):
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created output directory: {path}")

def validate_paths(data_path, output_path):
    """Ensure that the data path exists and the output directory is created."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")
    create_output_directory(output_path)

def save_losses(losses, filepath):
    """Save losses to a file."""
    np.savetxt(filepath, losses, fmt='%f')

def load_losses(filepath):
    """Load losses from a file."""
    return np.loadtxt(filepath)

def save_model(model, filepath):
    """Save the model to a file."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath):
    """Load the model from a file."""
    model.load_state_dict(torch.load(filepath))
    print(f"Model loaded from {filepath}")
    return model

def save_results(outputs, test_y, model, target, output_path, eval_mode, loss):
    """Save the evaluation results."""
    # Create target-specific results directory
    results_dir = os.path.join(output_path, f"{target}_results")
    create_output_directory(results_dir)

    # Save outputs and model
    torch.save(outputs, os.path.join(results_dir, f"{target}_test_results.pt"))
    torch.save(test_y, os.path.join(results_dir, f"{target}_test_target.pt"))
    save_model(model, os.path.join(results_dir, f"{target}_model.pth"))
    
    # Save test loss
    np.savetxt(os.path.join(results_dir, f"{target}_test_loss.txt"), [loss], fmt='%f')

    print(f"\nResults saved for {target} using {eval_mode} evaluation.")
    print(f"Test Loss: {loss:.6f}")

def get_evaluation_mode(target):
    """Get the evaluation mode for a given target."""
    from config import EVAL_MODES
    return EVAL_MODES.get(target, "non_sequential")
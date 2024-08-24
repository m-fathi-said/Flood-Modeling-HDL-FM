cd "C:\\Users\\msaid\\OneDrive - Villanova University\\Before Nov 2023\\Desktop\\Desktop\\Hydrodynamic ML\\Training dataset\\01 Final hydrodynamic paper\\Official code\\trail01 - Final"


import os
import config
from data_load import load_data
from model import create_model
from train_model import train_model
from test_model import run_test
from utils import validate_paths, set_random_seeds


def main():
    # Validate paths and create output directory if necessary
    validate_paths(config.DATA_PATH, config.OUTPUT_PATH)

    # Simplified target from config file
    target = config.TARGET
    if target not in config.TARGETS:
        raise ValueError(f"Invalid target specified: {target}. Please choose from {config.TARGETS}.")

    # Set random seed
    set_random_seeds(config.SEED)

    # Load data
    train_x, train_y, test_x, test_y = load_data(config.DATA_PATH, target)

    # Create model
    model = create_model(config.INPUT_CHANNELS, config.CNN1_CHANNELS, config.CNN2_CHANNELS, config.LSTM_INPUT_SIZE,
                         config.LSTM_HIDDEN_SIZE, config.LSTM_NUM_LAYERS, config.GRID_HEIGHT, config.GRID_WIDTH)

    # Train model
    train_model(model, train_x, train_y, config.EPOCHS, config.LEARNING_RATE, config.SMOOTH_L1_BETA, config.SCHEDULAR_F, config.SCHEDULAR_P, config.DEVICE)

    # Test model
    eval_mode = config.EVAL_MODES[target]
    run_test(model, test_x, test_y, target, config.SMOOTH_L1_BETA, eval_mode, config.OUTPUT_PATH)

if __name__ == "__main__":
    main()
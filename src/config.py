# User-defined paths
DATA_PATH = "path/to/your/data"  # User should modify this
OUTPUT_PATH = "path/to/output"   # User should modify this

# Grid size
GRID_HEIGHT = 520
GRID_WIDTH = 320

# Model parameters
INPUT_CHANNELS = 3
CNN1_CHANNELS = 13
CNN2_CHANNELS = 5
LSTM_INPUT_SIZE = 2600
LSTM_HIDDEN_SIZE = 2600
LSTM_NUM_LAYERS = 1

# Training parameters
EPOCHS = 1600
LEARNING_RATE = 0.00075
SMOOTH_L1_BETA = 0.1
SCHEDULAR_F = 0.85         # Scheduler factor
SCHEDULAR_P = 15           # Scheduler patience

# Other parameters
DEVICE = "cpu"             # "cpu" or "cuda"
SEED = 10

# Target options
TARGETS = ["water_depth", "velocity_magnitude", "flow_direction"]
TARGET = "water_depth"     # Select one Target

# Evaluation modes
EVAL_MODES = {
    "water_depth": "sequential",
    "velocity_magnitude": "non_sequential",
    "flow_direction": "non_sequential"
}

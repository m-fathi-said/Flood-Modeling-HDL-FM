import torch

def load_data(data_path, target):
    """Load training and testing data from the specified path."""
    train_x = torch.load(f"{data_path}/train_x.pt")
    train_y = torch.load(f"{data_path}/train_y.pt")
    test_x = torch.load(f"{data_path}/test_x.pt")
    test_y = torch.load(f"{data_path}/test_y.pt")
    
    # Select channels based on the target variable
    train_x = select_channels(train_x, target)
    test_x = select_channels(test_x, target)
    
    # Select appropriate target
    train_y = select_target(train_y, target)
    test_y = select_target(test_y, target)
    
    return train_x, train_y, test_x, test_y

def select_channels(x, target):
    channel_map = {
        "water_depth": [0, 1, 2],
        "velocity_magnitude": [0, 1, 3],
        "flow_direction": [0, 1, 3]
    }
    return x[:, channel_map[target], :, :]

def select_target(y, target):
    target_map = {
        "water_depth": 0,
        "velocity_magnitude": 1,
        "flow_direction": 2
    }
    return y[:, target_map[target], :, :].unsqueeze(1)
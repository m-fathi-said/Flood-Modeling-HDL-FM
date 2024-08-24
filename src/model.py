import torch
import torch.nn as nn

class HydrologicalModel(nn.Module):
    def __init__(self, input_channels, cnn1_channels, cnn2_channels, lstm_input_size, lstm_hidden_size, lstm_num_layers, H, W):
        super(HydrologicalModel, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, cnn1_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(cnn1_channels, cnn2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(cnn2_channels, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_input_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)
        
        # Linear layer
        self.linear = nn.Linear(lstm_hidden_size, H*W)
        

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size = x.size(0)
        
        # CNN
        x = self.cnn(x)
        
        # Flatten
        x = self.flatten(x)
        
        # LSTM
        x, _ = self.lstm(x.unsqueeze(1))  # Add sequence dimension
        #x = x.squeeze(1)  # Remove sequence dimension
        
        # Linear
        x = self.linear(x)
        
        return x

def create_model(input_channels, cnn1_channels, cnn2_channels, lstm_input_size, lstm_hidden_size, lstm_num_layers, H, W):
    model = HydrologicalModel(input_channels, cnn1_channels, cnn2_channels, lstm_input_size, lstm_hidden_size, lstm_num_layers, H, W)
    return model
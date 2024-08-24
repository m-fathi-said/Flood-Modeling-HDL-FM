import torch
from tqdm import tqdm

def train_epoch(model, train_x, train_y, criterion, optimizer, scheduler, device):
    model.train()
    output_shape = train_y.shape

    inputs, targets = train_x, train_y
    inputs, targets = inputs.to(device), targets.to(device)
    
    optimizer.zero_grad()
    outputs = model(inputs)
    outputs = outputs.view(*output_shape)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    return loss

def train_model(model, train_x, train_y, epochs, learning_rate, smooth_l1_beta, scheduler_factor, scheduler_patience, device):
    """Train the model."""
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = torch.nn.SmoothL1Loss(beta=smooth_l1_beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.85, patience=15, verbose=True)

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        train_loss = train_epoch(model, train_x, train_y, criterion, optimizer, scheduler, device)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}')
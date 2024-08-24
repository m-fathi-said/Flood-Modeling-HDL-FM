import torch
import numpy as np
from tqdm import tqdm
from utils import save_results

def test_sequential(model, test_x, test_y, criterion):
    """Testing the model using sequential prediction for water depth."""
    model.eval()
    device = next(model.parameters()).device
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    
    outputs = torch.zeros_like(test_y)
    test_x_clone = test_x.clone()
    output_shape = (1, 1, test_x.shape[2], test_x.shape[3])
    
    with torch.no_grad():
        for i in tqdm(range(test_y.shape[0]-1), desc="Sequential Evaluation"):
            one_x = test_x_clone[i,:,:,:] # step by step
            output = model(one_x)
            output = output.view(*output_shape)
            test_x_clone[i+1,2,:,:] = output
            outputs[i] = output

    loss = criterion(outputs, test_y)
    return outputs.cpu(), loss.item()

def test_non_sequential(model, test_x, test_y, criterion):
    """Testing the model using non-sequential prediction for velocity magnitude and flow direction."""
    model.eval()
    device = next(model.parameters()).device
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    output_shape = test_y.shape
    
    with torch.no_grad():
        outputs = model(test_x)
        outputs = outputs.view(*output_shape)
        loss = criterion(outputs, test_y)
    
    return outputs.cpu(), loss.item()

def test_model(model, test_x, test_y, criterion, mode='sequential'):
    """Testing the model using either sequential or non-sequential prediction."""
    if mode == 'sequential':
        return test_sequential(model, test_x, test_y, criterion)
    elif mode == 'non_sequential':
        return test_non_sequential(model, test_x, test_y, criterion)
    else:
        raise ValueError("Mode must be either 'sequential' or 'non_sequential'")

def run_test(model, test_x, test_y, target, smooth_l1_beta, eval_mode, output_path):
    """Run the full evaluation process and save results."""
    criterion = torch.nn.SmoothL1Loss(beta=smooth_l1_beta)
    outputs, loss = test_model(model, test_x, test_y, criterion, eval_mode)
    save_results(outputs, test_y, model, target, output_path, eval_mode, loss)
# Hydrodynamic Hybrid Deep-Learning for Flood Modeling HDL-FM
This code supports the research presented in the paper "Spatiotemporal flood depth and velocity dynamics using a convolutional neural network within a sequential Deep-Learning framework" (Environmental Modelling & Software, https://doi.org/10.1016/j.envsoft.2024.106307). 


This repository contains preprocessed input-output pairs for training and testing of a hybrid deep learning architecture designed to simulate hydrodynamic flood dynamics. The framework integrates Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks to model temporal dependencies in flood evolution. The dataset, hosted on Zenodo (https://zenodo.org/records/15223719), includes four PyTorch-compatible .pt files:

train_x.pt: Model inputs (e.g., topography, discharge)
train_y.pt: Model targets (water depth, velocity magnitude, and flow direction)
test_x.pt: Same inputs for testing part
test_y.pt: Same targets for testing part

Recommended Citation
Fathi, M.M., Liu, Z., Fernandes, A.M., Hren, M.T., Terry, D.O., Nataraj, C. and Smith, V., 2025. Spatiotemporal flood depth and velocity dynamics using a convolutional neural network within a sequential Deep-Learning framework. Environmental Modelling & Software, 185, p.106307.

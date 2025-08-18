import numpy as np
import seaborn as sns
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))
])

def load_data():
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data  = datasets.MNIST('./data', train=False, transform=transform) 

    train_size = int(0.6 * len(train_data))
    val_size = len(train_data) - train_size
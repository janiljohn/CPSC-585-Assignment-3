import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)


print('PyTorch version:', torch.__version__)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import gc
from sklearn.metrics import mean_squared_error

from physical_context_features import feature_calculator
from process_data import *


# set parameters

from sklearn.model_selection import KFold


def extract_batch_features(feat,batch_size):
    context_tensor=None
    # loop through the batches and apply the extract_feature function on each element
    for i in range(batch_size):
        batch_element = feat[i]

        # convert the PyTorch tensor to a numpy array
        batch_element_np = batch_element.numpy()

        # apply the extract_feature function on the numpy array
        feature_np = feature_calculator(batch_element_np)

        # convert the resulting feature back to a PyTorch tensor
        feature = torch.from_numpy(feature_np).float()

        # create a new context tensor by concatenating the previous tensor with the new feature tensor
        if i == 0:
            context_tensor = feature.unsqueeze(0)
        else:
            context_tensor = torch.cat((context_tensor, feature.unsqueeze(0)), dim=0)

    return context_tensor
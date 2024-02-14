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
from deepl_data_loader import *

# set parameters
from evaluate import evaluate_model
from tafi_model_10 import TAFIWithFCOutput as TAFI
from sklearn.model_selection import KFold


def load_model(model, filepath):
    device = torch.device('cpu')
    model.load_state_dict(torch.load(filepath,map_location=device))


spatial_input_channels = 6
spatial_output_channels = 64
#
temporal_input_size = 200
temporal_hidden_size = 100
temporal_num_layers = 1
#
encoder_input_size = 64
encoder_output_size = 128

decoder_input_size = 200

batch_size= 8 # batch size, amount of data to be used in a batch
dropout = 0.5 # dropout rate
step_size = 200 # how many steps the sliding window is moving
window_size = 200 # window size of the sliding window

folders = glob.glob("./data/Data RIDI test/**/processed/")
#folders = glob.glob("./data/Data OXIOD test/**/**/")
#folders = glob.glob("./data/IIS test/**/Sync/*csv")

r = 0
for folder in folders:
    feature_size = 25
    model = TAFI(context_size=feature_size, num_heads=8, num_layers=2, hidden_size=256, dropout=0.1)

    load_model(model,"models/tafi/ridi/model_244.pth" )
    model.eval()
    print(f"folder is {folder}")
    x_test,y_test = data_loader(step_size=step_size, window_size=window_size, type='one',path=folder).load_ridi()
    x_data_test = torch.tensor(x_test).float()
    y_data_test = torch.tensor(y_test).float()
    data_set_test = TensorDataset(x_data_test,y_data_test)
    train_batches_test = DataLoader(data_set_test, batch_size=batch_size, shuffle=False,drop_last=True)
    criterion = nn.MSELoss()
    evaluate_model(model,train_batches_test, None,path="predictions/tafi/ridi/"+str(r),window_size=window_size,criterion=criterion,gpu=False,plot=True, batch_size=batch_size)
    r+=1

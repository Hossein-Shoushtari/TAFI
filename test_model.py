import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import gc
from sklearn.metrics import mean_squared_error

from Physical_context_features import feature_calculator
from ml_loader import extract_features
from process_data import *
from nn_models import *
from deepl_data_loader import *

# set parameters
from training_scripts.evaluate import evaluate_model
from new_transformer_model import TAFI as CTIN_with_features
from sklearn.model_selection import KFold


def load_model(model, filepath):
    device = torch.device('cpu')
    model.load_state_dict(torch.load(filepath,map_location=device))


batch_size= 128 # batch size, amount of data to be used in a batch
dropout = 0.5 # dropout rate
step_size = 10 # how many steps the sliding window is moving
window_size = 200 # window size of the sliding window

folders = glob.glob("./data/Data RIDI test/**/processed/")
counter = 0
for folder in folders:
    feature_size = 25
    model = CTIN_with_features(feature_size)

    load_model(model,"TAFI_on_ronin.pth" )
    model.eval()
    print(f"folder is {folder}")
    x_test,y_test = data_loader(step_size=step_size, window_size=window_size, type='one',path=folder).load_ronin()
    x_data_test = torch.tensor(x_test).float()
    y_data_test = torch.tensor(y_test).float()
    data_set_test = TensorDataset(x_data_test,y_data_test)
    train_batches_test = DataLoader(data_set_test, batch_size=batch_size, shuffle=False,drop_last=True)
    criterion = nn.MSELoss()
    evaluate_model(model,train_batches_test, None,path="predictions/"+counter,window_size=window_size,criterion=criterion,gpu=False,plot=True)
    counter+=1 

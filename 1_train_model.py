import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import gc
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from torch.optim.lr_scheduler import ReduceLROnPlateau
from physical_context_features import feature_calculator
#from ml_loader import extract_features
from process_data import *
#from nn_models import *
from deepl_data_loader import *

# set parameters
from evaluate import evaluate_model
from utils import extract_batch_features
from tafi_model_10 import TAFIWithFCOutput as TAFI
#from CTIN_model import CTIN as CTIN_with_features
from sklearn.model_selection import KFold

# Define hyperparameters
NUM_FOLDS = 5
set_seed(1337)
EPOCHS = 250 # number of epochs / training cycles

learning_rate = 0.0001  # learning rate of the model
batch_size= 256 # batch size, amount of data to be used in a batch
dropout = 0.5 # dropout rate
step_size = 10 # how many steps the sliding window is moving
window_size = 200 # window size of the sliding window
#writer = SummaryWriter(log_dir=f'./runs/ResNet and OX')
#%%
Epoch_start = 0

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))


#epoch121 LR: 0.000810, Loss: 0.1549  Dev loss: 0.1845 RMSE : 0.3572

def train_model(model, train_batches, epoch, optimizer, scheduler=None, history=None,cuda=True):
    model.train()
    total_loss = 0
    t = tqdm(train_batches)

    for i,(feat, targ) in enumerate(t):


        optimizer.zero_grad()

        context_tensor = extract_batch_features(feat, batch_size)
        if cuda:
            context_tensor = context_tensor.to('cuda')
            feat, targ = feat.to('cuda'), targ.to('cuda')

        feat = feat.reshape(feat.shape[0], 6, window_size)
        #context_tensor = None
        out = model(feat,context_tensor)

        loss = criterion(out, targ)

        total_loss += loss.item()
        #writer.add_scalar("Loss/train", total_loss/(i+1), epoch)

        t.set_description(f'Epoch {epoch+1} : , LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(i+1)))

        loss.backward()
        optimizer.step()
    history['train_loss'].append(total_loss / len(train_batches))


def plot_loss(name,train_loss, val_loss):
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss'+name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('models/tafi/ridi/train_vs_loss'+name+'.png')
    plt.show()




#wrap_data_instance = wrap_data(step_size=step_size, window_size=window_size, batch_size= batch_size)
#x_train,y_train = wrap_data_instance.load_training_dataset()
#x_test,y_test = wrap_data_instance.load_testing_dataset()
#X,y, = data_loader(step_size=step_size, window_size=window_size, type='train').load_oxiod()
#x_test,y_test, = data_loader(step_size=step_size, window_size=window_size, type='test').load_oxiod()

#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



X,y, = data_loader(step_size=step_size, window_size=window_size, type='train').load_ridi()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#x_train,y_train, = data_loader(step_size=step_size, window_size=window_size, type='train').load_ridi()
#x_test,y_test, = data_loader(step_size=step_size, window_size=window_size, type='test').load_ridi()



feature_size = 25
model = TAFI(context_size=feature_size)#CTIN_with_features(context_size=feature_size)#CTIN(context_size=feature_size)
#load_model(model, f"Transformer_p_on_ridi/model_{Epoch_start}.pth")

cuda=True

if cuda:
    model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.9, verbose=True, eps=1e-8)

x_data = torch.tensor(x_train).float()
y_data = torch.tensor(y_train).float()
x_data_test = torch.tensor(x_test).float()
y_data_test = torch.tensor(y_test).float()
del x_train,y_train,x_test,y_test
gc.collect()

# wrap data into the dataloader
data_set = TensorDataset(x_data,y_data)
train_batches = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=True)
data_set_test = TensorDataset(x_data_test,y_data_test)
train_batches_test = DataLoader(data_set_test, batch_size=batch_size, shuffle=False, drop_last=True)
del x_data,y_data,x_data_test,y_data_test




best_loss=10
history = {'train_loss': [], 'val_loss': []}
# train model
for epoch in range(Epoch_start, EPOCHS):
    train_model(model, train_batches, epoch, optimizer, scheduler=scheduler, history=history,cuda=cuda)
    val_loss= evaluate_model(model, train_batches_test, epoch, scheduler=scheduler, history=history,window_size=window_size, batch_size=batch_size,criterion = criterion,gpu=cuda)
    history['val_loss'].append(val_loss.item())
    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f'./models/tafi/ridi/model_{epoch}.pth')
        plot_loss("_" + str(epoch), history['train_loss'],history['val_loss'])
plot_loss("final plot",history['train_loss'],history['val_loss'])


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
from utils import extract_batch_features

from sklearn.model_selection import KFold

def plot_trajectories(vx, vy, pred_vx, pred_vy):
    dt = 0.1  # Time step for integration

    # Integrate velocities to obtain positions
    x = np.cumsum(vx) * dt
    y = np.cumsum(vy) * dt

    pred_x = np.cumsum(pred_vx) * dt
    pred_y = np.cumsum(pred_vy) * dt

    # Plot original trajectory
    plt.plot(x, y, 'b', label='Original Trajectory')

    # Plot predicted trajectory
    plt.plot(pred_x, pred_y, 'r', label='Predicted Trajectory')

    # Set labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory Comparison')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()

def evaluate_model(model, train_batches_test, epoch, scheduler=None, history=None,path=None,window_size=None,batch_size=8,criterion=None,plot=False,gpu=True):
    model.eval()
    total_loss = 0
    global pred_list
    pred_list  = []
    global real_list
    real_list = []
    global RMSE_list
    RMSE_list = []
    with torch.no_grad():
        for i,(feat, targ) in enumerate(tqdm(train_batches_test)):
            context_tensor = extract_batch_features(feat, batch_size)

            if gpu:
                context_tensor = context_tensor.to('cuda')
                feat, targ = feat.to('cuda'), targ.to('cuda')

            feat = feat.reshape(feat.shape[0], 6, window_size)
            #context_tensor = None
            out = model(feat,context_tensor)
            loss = criterion(out, targ)
            total_loss += loss

            out = out.cpu().numpy()
            targ = targ.cpu().numpy()

            for pred, real in zip(out, targ):
                rmse = np.sqrt(mean_squared_error(real, pred))
                RMSE_list.append(rmse)
                pred_list.append(pred)
                real_list.append(real)

    total_loss /= len(train_batches_test)

    if plot:
        # Save predictions and targets to csv file
        df = pd.DataFrame({'Prediction': pred_list, 'Target': real_list})
        print("saving to"+path+'.csv')
        df.to_csv(path+'.csv', index=False)
        plot_trajectories([item[0] for item in real_list],[item[1] for item in real_list],[item[0] for item in pred_list],[item[1] for item in pred_list])


    if scheduler is not None:
        with torch.no_grad():
            scheduler.step(total_loss)

    #writer.add_scalar("Loss/test", total_loss/(i+1), epoch)
    print(f'\n Dev loss: %.4f RMSE : %.4f'%(total_loss, np.mean(RMSE_list)))
    return total_loss

def evaluate_one_input_model(model, train_batches_test, epoch, scheduler=None, history=None,path=None,window_size=None,batch_size=8,criterion=None,plot=False,gpu=True):
    model.eval()
    total_loss = 0
    global pred_list
    pred_list  = []
    global real_list
    real_list = []
    global RMSE_list
    RMSE_list = []
    with torch.no_grad():
        for i,(feat, targ) in enumerate(tqdm(train_batches_test)):

            if gpu:
                feat, targ = feat.to('cuda'), targ.to('cuda')

            feat = feat.reshape(feat.shape[0], 6, window_size)
            out,_ = model(feat)
            loss = criterion(out, targ)
            total_loss += loss

            out = out.cpu().numpy()
            targ = targ.cpu().numpy()

            for pred, real in zip(out, targ):
                rmse = np.sqrt(mean_squared_error(real, pred))
                RMSE_list.append(rmse)
                pred_list.append(pred)
                real_list.append(real)

    total_loss /= len(train_batches_test)

    if plot:
        # Save predictions and targets to csv file
        df = pd.DataFrame({'Prediction': pred_list, 'Target': real_list})
        print("saving to"+path+'.csv')
        df.to_csv(path+'.csv', index=False)
        plot_trajectories([item[0] for item in real_list],[item[1] for item in real_list],[item[0] for item in pred_list],[item[1] for item in pred_list])


    if scheduler is not None:
        with torch.no_grad():
            scheduler.step(total_loss)

    #writer.add_scalar("Loss/test", total_loss/(i+1), epoch)
    print(f'\n Dev loss: %.4f RMSE : %.4f'%(total_loss, np.mean(RMSE_list)))
    return total_loss


def evaluate_resnet_model(model, train_batches_test, epoch, scheduler=None, history=None,path=None,window_size=None,batch_size=8,criterion=None,plot=False,gpu=True):
    model.eval()
    total_loss = 0
    global pred_list
    pred_list  = []
    global real_list
    real_list = []
    global RMSE_list
    RMSE_list = []
    with torch.no_grad():
        for i,(feat, targ) in enumerate(tqdm(train_batches_test)):


            if gpu:
                context_tensor = context_tensor.to('cuda')
                feat, targ = feat.to('cuda'), targ.to('cuda')

            feat = feat.reshape(feat.shape[0], 6, window_size)
            # extract features and concat
            context_tensor = extract_batch_features(feat, batch_size)
            context_tensor = context_tensor.numpy()
            context_tensor = context_tensor.reshape(context_tensor.shape[0], 1, 25)
            repeated_context_tensor = np.tile(context_tensor, (1, 200, 1))
            context_tensor = torch.tensor(repeated_context_tensor).float()
            context_tensor = context_tensor.reshape(context_tensor.shape[0], 25, -1)
            concatenated_tensor = torch.cat([feat, context_tensor], dim=1)
            out = model(concatenated_tensor)
            loss = criterion(out, targ)
            total_loss += loss

            out = out.cpu().numpy()
            targ = targ.cpu().numpy()

            for pred, real in zip(out, targ):
                rmse = np.sqrt(mean_squared_error(real, pred))
                RMSE_list.append(rmse)
                pred_list.append(pred)
                real_list.append(real)

    total_loss /= len(train_batches_test)

    if plot:
        # Save predictions and targets to csv file
        df = pd.DataFrame({'Prediction': pred_list, 'Target': real_list})
        print("saving to"+path+'.csv')
        df.to_csv(path+'.csv', index=False)
        plot_trajectories([item[0] for item in real_list],[item[1] for item in real_list],[item[0] for item in pred_list],[item[1] for item in pred_list])


    if scheduler is not None:
        with torch.no_grad():
            scheduler.step(total_loss)

    #writer.add_scalar("Loss/test", total_loss/(i+1), epoch)
    print(f'\n Dev loss: %.4f RMSE : %.4f'%(total_loss, np.mean(RMSE_list)))
    return total_loss



def evaluate_resnet_model2(model, train_batches_test, epoch, scheduler=None, history=None,path=None,window_size=None,batch_size=8,criterion=None,plot=False,gpu=True):
    model.eval()
    total_loss = 0
    global pred_list
    pred_list  = []
    global real_list
    real_list = []
    global RMSE_list
    RMSE_list = []
    with torch.no_grad():
        for i,(feat, targ) in enumerate(tqdm(train_batches_test)):


            if gpu:
                context_tensor = context_tensor.to('cuda')
                feat, targ = feat.to('cuda'), targ.to('cuda')

            feat = feat.reshape(feat.shape[0], 6, window_size)
            # extract features and concat
            #context_tensor = extract_batch_features(feat, batch_size)
            #context_tensor = context_tensor.numpy()
            #context_tensor = context_tensor.reshape(context_tensor.shape[0], 1, 25)
            #repeated_context_tensor = np.tile(context_tensor, (1, 200, 1))
            #context_tensor = torch.tensor(repeated_context_tensor).float()
            #context_tensor = context_tensor.reshape(context_tensor.shape[0], 25, -1)
            #concatenated_tensor = torch.cat([feat, context_tensor], dim=1)
            out = model(feat)
            loss = criterion(out, targ)
            total_loss += loss

            out = out.cpu().numpy()
            targ = targ.cpu().numpy()

            for pred, real in zip(out, targ):
                rmse = np.sqrt(mean_squared_error(real, pred))
                RMSE_list.append(rmse)
                pred_list.append(pred)
                real_list.append(real)

    total_loss /= len(train_batches_test)

    if plot:
        # Save predictions and targets to csv file
        df = pd.DataFrame({'Prediction': pred_list, 'Target': real_list})
        print("saving to"+path+'.csv')
        df.to_csv(path+'.csv', index=False)
        plot_trajectories([item[0] for item in real_list],[item[1] for item in real_list],[item[0] for item in pred_list],[item[1] for item in pred_list])


    if scheduler is not None:
        with torch.no_grad():
            scheduler.step(total_loss)

    #writer.add_scalar("Loss/test", total_loss/(i+1), epoch)
    print(f'\n Dev loss: %.4f RMSE : %.4f'%(total_loss, np.mean(RMSE_list)))
    return total_loss



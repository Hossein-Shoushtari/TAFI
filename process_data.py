# Process the loaded data and put it into the DataLoader

import numpy as np
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
from deepl_data_loader import *

class wrap_data:
    def __init__(self, step_size: int, window_size: int, batch_size: int):
        self.step_size = step_size
        self.window_size = window_size
        self.batch_size = batch_size
       
    def load_training_dataset(self):
        print("loading training data")
        X1,y1, = data_loader(step_size=self.step_size, window_size=self.window_size, type='train').load_ronin()
        X2,y2 = data_loader(step_size=self.step_size, window_size=self.window_size, type='train').load_ridi()
        X3,y3 = data_loader(step_size=self.step_size, window_size=self.window_size, type='train').load_oxiod()
        X4,y4 = data_loader(step_size=self.step_size, window_size=self.window_size, type='train').load_iis()
        return np.concatenate((X1,X2,X3, X4), axis=0), np.concatenate((y1,y2,y3, y4), axis=0)

    def load_testing_dataset(self):
        print("loading testing data")
        X1,y1 = data_loader(step_size=self.step_size, window_size=self.window_size, type='test').load_ronin()
        X2,y2 = data_loader(step_size=self.step_size, window_size=self.window_size, type='test').load_ridi()
        X3,y3 = data_loader(step_size=self.step_size, window_size=self.window_size, type='test').load_oxiod()
        X4,y4 = data_loader(step_size=self.step_size, window_size=self.window_size, type='test').load_iis()
        return np.concatenate((X1,X2, X3, X4), axis=0), np.concatenate((y1,y2, y3, y4), axis=0)

    def load_validation_dataset(self):
        print("loading validation data")
        X1,y1 = data_loader(step_size=self.step_size, window_size=self.window_size, type='val').load_ronin()
        #X2,y2 = data_loader(step_size=self.step_size, window_size=self.window_size, type='val').load_ridi()
        X3,y3 = data_loader(step_size=self.step_size, window_size=self.window_size, type='val').load_oxiod()
        #X4,y4 = data_loader(step_size=self.step_size, window_size=self.window_size, type='val').load_iis()
        return np.concatenate((X1, X3), axis=0), np.concatenate((y1, y3), axis=0)


    # transform data into tensorsf
    def data_to_tensor(self):
        """Transform concatenated data into tensors

        Returns:
            array: transformed features, labels into tensors
        """

        X_train, y_train = self.load_training_dataset()
        X_test, y_test = self.load_testing_dataset()

        print("transforming data into tensors")
        
        x_data = torch.tensor(X_train).float()
        del X_train
        y_data = torch.tensor(y_train).float()
        del y_train
        x_data_test = torch.tensor(X_test).float()
        del X_test
        y_data_test = torch.tensor(y_test).float()
        del y_test
        
        return x_data, y_data, x_data_test, y_data_test


    # wrap data into the dataloader
    def load_preprocess_wrap_data(self):
        """Wrap tranformed tensors into the dataloader

        Returns:
            DataLoader: Preprocessed data and wrapped into a DataLoader
        """
        x_data, y_data, x_data_test, y_data_test = self.data_to_tensor()
        
        print("wraping data into dataloader")
        
        data_set = TensorDataset(x_data,y_data)
        del [x_data, y_data]
        train_batches = DataLoader(data_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        data_set_test = TensorDataset(x_data_test,y_data_test)
        del [x_data_test, y_data_test]
        train_batches_test = DataLoader(data_set_test, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        return train_batches, train_batches_test


def set_seed(seed):
    """Set fixed seed for reproducibility

    Args:
        seed (int): Seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

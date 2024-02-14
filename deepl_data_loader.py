# Load RONIN and RIDI data, later OXIOD

import numpy as np
import json
import h5py
import glob
import quaternion
import gc
import os
import pandas as pd
from scipy.spatial.transform import Rotation
#from init_orientation import initial_tango_orientation

class data_loader:
    """
    Function that loads one dataset.
    
    Args:
        steop_size (int): amount of steps between two consecutive windows
        window_size (int): amount of data within a window
        type (str): type of dataset to be used
        
        Returns:
            numpy arrays: features, labels"""

    def __init__(self, step_size: int, window_size: int, type: str,path=None):
        self.step_size = step_size
        self.window_size = window_size
        self.type = type
        self.path = path

# ------------------------------------------------RONIN----------------------------------------------------------------------
    def load_ronin(self):
        """
        Load the RONIN dataset.
        
        
        Returns:
            numpy arrays: features, labels"""

        list_of_features,list_of_labels, list_for_all_ronin = [], [], []

        if self.type == 'test':
            folders = glob.glob('./data/Data ronin test/**/**/') 
        if self.type == 'train':
            folders = glob.glob('./data/Data ronin/**/**/')
        if self.type == 'val':
            folders = glob.glob('./data/Data ronin val/**/**/')
        if self.type == 'one':
            folders = glob.glob(self.path)


        for folder in folders:
            for f in glob.glob(folder + '/*hdf5'):
                list_for_all_ronin.append(f)

        # recursive operation for every file
        for file_path in (list_for_all_ronin):
            with h5py.File(file_path) as f:
                info = json.load(open(file_path[:-9]+ 'info.json'))
                gyro_uncalib = f['synced/gyro_uncalib']
                acce_uncalib = f['synced/acce']
                gyro = gyro_uncalib - np.array(info['imu_init_gyro_bias'])
                acce = np.array(info['imu_acce_scale']) * (acce_uncalib - np.array(info['imu_acce_bias']))
                ts = np.copy(f['synced/time'])
                tango_pos = np.copy(f['pose/tango_pos']) 
                init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])
                
            info['ori_source'],ori,info['source_ori_error'] = data_loader.select_orientation_source(file_path[:-9])
            
            # Compute the IMU orientation in the Tango coordinate frame.
            ori_q = quaternion.from_float_array(ori)
            rot_imu_to_tango = quaternion.quaternion(*info['start_calibration'])
            init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
            ori_q = init_rotor * ori_q

            gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
            acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
            glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
            glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
            start_frame = info.get('start_frame', 0)
            ts = ts[start_frame:]
            features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:]
            y = tango_pos[start_frame:]
            
            # garbage collection
            del [info, gyro_uncalib, acce_uncalib, gyro, acce, tango_pos, init_tango_ori, ori_q, gyro_q, acce_q, glob_gyro, glob_acce, start_frame]
            gc.collect()

            # sliding window with overlap to calculate features and labels
            #y=17ß000
            #diff_conc = 11000
            difference_conc = self.sliding_window_label(y, step_size=self.step_size, window_size=self.window_size)
            final_features = self.sliding_window_features(features, step_size=self.step_size, window_size=self.window_size)
            #feature: 35300
            #final feature: 7000000
            list_of_labels.append(difference_conc)
            del [features, y, ts, difference_conc]
            list_of_features.append(final_features)   
          
        # concatenate all features and labels
        X = np.concatenate(list_of_features)
        del list_of_features
        y = np.concatenate(list_of_labels)
        del list_of_labels

        return X, y


# ------------------------------------------------RIDI----------------------------------------------------------------------
# Loading the RIDI data is split into multiple parts beacuse of problems with memory

    def ridi_preprocess(self, interval_start: int, interval_end: int):
        """
        Preprocess the RIDI dataset.
        
        Args:
            interval_start (int): amount of files to be used
            interval_end (int): amount of files to be used

        Returns:
            numpy array: lists of features, labels and plot
        """
        # define path
        if self.type == 'train':
            path = './data/Data RIDI/**/**/processed/'
        if self.type == 'test': 
            path = './data/Data RIDI test/**/processed/'
        if self.type == 'one':
            path=self.path
            
        feature_filenames = glob.glob(path + 'data.csv')
    
        # generate two empty lists for features and labels
        list_of_features, list_of_labels = [], []
        
        for filename in feature_filenames[interval_start:interval_end]:
            
            # first features are processed
            imu_all = pd.read_csv(filename, low_memory=False) 
            gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
            acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
            tango_pos = imu_all[['pos_x', 'pos_y', 'pos_z']].values
            
            # preproces data 
            ts = imu_all[['time']].values / 1e09
            init_tango_ori = quaternion.quaternion(*imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values[0])
            game_rv =  quaternion.from_float_array(imu_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)
            init_rotor = init_tango_ori * game_rv[0].conj()
            ori = init_rotor * game_rv
            
            nz = np.zeros(ts.shape)
            nz = np.reshape(nz, (nz.shape[0], 1))
            gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
            acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))

            gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
            acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]
            
            # append features to list
            features = np.concatenate([gyro_glob, acce_glob], axis=1)
            
            # garbage collection
            del [init_tango_ori, game_rv, init_rotor, ori, nz, gyro_q, acce_q, gyro_glob, acce_glob]
            gc.collect()
            
            # sliding window with overlap to calculate features and labels
            difference_conc = self.sliding_window_label(tango_pos, step_size=self.step_size, window_size=self.window_size)
            final_features = self.sliding_window_features(features, step_size=self.step_size, window_size=self.window_size)
                
            list_of_labels.append(difference_conc)
            list_of_features.append(final_features)

        return list_of_features, list_of_labels


    # function that concatenates the output of the load_ridi function
    def concat_ridi(self, interval_start: int, interval_end: int):
        """
        Concatenate the RIDI dataset.
        
        Args:
            interval_start (int): amount of files to be used
            interval_end (int): amount of files to be used
            overlap (bool): whether to use overlapping windows or not

        Returns:
            numpy arrays: lists of features, labels
        """
        X,y = self.ridi_preprocess(interval_start, interval_end)
        X_concat = np.concatenate(X, axis=0)
        y_concat = np.concatenate(y, axis=0)
        
        return X_concat, y_concat


    def load_ridi(self):
        """
        Load the RIDI training dataset.
        
        Args:
            interval_start (int): amount of files to be used
            interval_end (int): amount of files to be used

        Returns:
            numpy arrays: features, labels
        """
        
        if self.type == 'train':
            X_1, y_1 = self.concat_ridi(interval_start=0,interval_end=38)
            X_2, y_2 = self.concat_ridi(interval_start=39,interval_end=73)
        elif self.type == 'one':
            X_1, y_1 = self.concat_ridi(interval_start=0, interval_end=38)
            return X_1, y_1
        else: 
            X_1, y_1 = self.concat_ridi(interval_start=0,interval_end=10)
            X_2, y_2 = self.concat_ridi(interval_start=11,interval_end=21)

        return np.concatenate([X_1, X_2], axis=0), np.concatenate([y_1, y_2], axis=0)


# ------------------------------------------------OXIOD----------------------------------------------------------------------
    # load OXIOD data
    def load_oxiod(self):
        """
        Load the OXIOD dataset.
        

        Returns:
            numpy arrays: features, labels
        """

        # define path
        if self.type == 'train':
            folders = glob.glob('./data/Data OXIOD/**/**/**/')
        if self.type == 'test':
            folders = glob.glob('./data/Data OXIOD test/**/**/')
        if self.type == 'val':
            folders = glob.glob('./data/Data OXIOD val/**/')
        if self.type == 'one':
            folders = glob.glob(self.path)
            
        # generate two empty lists for the directorys and two for features + labels
        list_of_features, list_of_labels = [], []
        ox_feature, ox_label = [], []

        # loop over all folders
        for folder in folders:
            for f in glob.glob(folder + 'imu*'):
                ox_feature.append(f)
            for g in glob.glob(folder + 'vi*'):
                ox_label.append(g)

        # sort the lists
        ox_feature.sort()
        ox_label.sort()

        # column names
        feature_columns = ['Time', 'attitude_roll', 'attitude_pitch', 'attitude_yaw', 
                    'rotation_rate_x', 'rotation_rate_y', 'rotation_rate_z', 
                    'gravity_x', 'gravity_y', 'gravity_z', 'user_acc_x', 'user_acc_y', 'user_acc_z', 
                    'magnetic_field_x', 'magnetic_field_y', 'magnetic_field_z']

        label_columns = ['Time', 'Header', 'translation.x', 'translation.y', 'translation.z',
                    'rotation.x', 'rotation.y', 'rotation.z', 'rotation.w']

        # loop to preprocess the data
        for i in range(len(ox_feature)):
            df = pd.read_csv(ox_feature[i], header=None, names=feature_columns, low_memory=False)
            df_2 = pd.read_csv(ox_label[i], header=None, names=label_columns, low_memory=False)

            gyro = df[['rotation_rate_x', 'rotation_rate_y', 'rotation_rate_z']].values
            acce = df[['user_acc_x', 'user_acc_y', 'user_acc_z']].values
            tango_pos = df_2[['translation.x', 'translation.y', 'translation.z']].values
        
            ts = df['Time'].values
            init_tango_ori = quaternion.quaternion(*df_2[['rotation.w', 'rotation.x', 'rotation.y', 'rotation.z']].values[0])
            #game_rv = get_quaternion_from_euler(df['attitude_roll'], df['attitude_pitch'], df['attitude_yaw']).reshape(-1,)
            game_rv = quat_from_rpy_loop(df['attitude_roll'], df['attitude_pitch'], df['attitude_yaw'])
            init_rotor = init_tango_ori * game_rv[0].conj()
            ori = init_rotor * game_rv

            nz = np.zeros(ts.shape)
            nz = np.reshape(nz, (nz.shape[0], 1))
            gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
            acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))

            gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
            acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]

            # append features to list
            features = np.concatenate([gyro_glob*10, acce_glob*10], axis=1)

            # garbage collection
            del [init_tango_ori, game_rv, init_rotor, ori, nz, gyro_q, acce_q, gyro_glob, acce_glob]
            gc.collect()

            # sliding window with overlap to calculate features and labels
            difference_conc = self.sliding_window_label(tango_pos, step_size=self.step_size, window_size=self.window_size)
            final_features = self.sliding_window_features(features, step_size=self.step_size, window_size=self.window_size)
                
            list_of_labels.append(difference_conc)
            del [features, ts, difference_conc]
            list_of_features.append(final_features) 

        X = np.concatenate(list_of_features)
        del list_of_features
        y = np.concatenate(list_of_labels)
        del list_of_labels

        return X, y


# ------------------------------------------------L5INP----------------------------------------------------------------------
    # load L5INP data
    def load_iis(self):
        """
        Load the L5INp dataset.
        
        
        Returns:
            numpy arrays: features, labels"""

        # path to the data
        if self.type == 'train':
            feature_filenames = glob.glob('./data/IIS train/**/**/Sync/*csv')
        if self.type == 'test':
            feature_filenames = glob.glob('./data/IIS test/**/Sync/*csv')
        if self.type == 'one':
            feature_filenames = glob.glob(self.path)


        # generate two empty lists for features and labels
        list_of_features, list_of_labels = [], []

        for filename in feature_filenames:

            imu_all = pd.read_csv(filename, low_memory=False)

            imu_all.dropna(subset=['gyro_x'], inplace=True)
            imu_all['time'] = pd.to_numeric(imu_all['time'], errors='coerce')
           
            
            gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
            acce = imu_all[['acc_x', 'acc_y', 'acc_z']].values
            tango_pos = imu_all[['x_pos_qsys1', 'y_pos_qsys1', 'z_pos_qsys1']].values

            for j in range(15,65,5):
                load_ori = initial_tango_orientation(filename, 100, 1, j) ########!!!!!!! j
                init_tango_ori = load_ori()
        
                game_rv = quaternion.from_float_array(imu_all[['attitude_w', 'attitude_x', 'attitude_y', 'attitude_z']].values)

                ts = imu_all[['time']].values /1e09
                init_rotor = init_tango_ori * game_rv[0].conj()
                ori =  init_rotor * game_rv
                nz = np.zeros(ts.shape)
                nz = np.reshape(nz, (nz.shape[0], 1))
                gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
                acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))

                gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
                acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]

                features = np.concatenate([gyro_glob, acce_glob], axis=1)

                if ( (sum(features[:,4])/len(features[:,4]) < 1 and sum(features[:,4])/len(features[:,4]) > -1 ) and
                    (sum(features[:,5])/len(features[:,5]) < 10 and sum(features[:,5])/len(features[:,5]) > 9 )):
                    break
            #features[:,3]  = features[:,3] - sum(features[:,3])/len(features[:,3])
            #features[:,4]  = features[:,4]  - sum(features[:,4])/len(features[:,4])

            # garbage collection
            del [init_tango_ori, game_rv, init_rotor, ori, nz, gyro_q, acce_q, gyro_glob, acce_glob]
            gc.collect()

            final_features, difference_conc = self.slider(features, tango_pos, step_size=self.step_size, window_size=self.window_size)


            list_of_nan_rows = []
            for k in range(len(difference_conc)):
                if np.any(np.isnan(difference_conc[k])) == True:
                    list_of_nan_rows.append(k)

            difference_conc = np.delete(difference_conc, list_of_nan_rows, axis=0)
            final_features = np.delete(final_features, list_of_nan_rows, axis=0)

           
            list_of_labels.append(difference_conc)
            del [features, ts, difference_conc]
            list_of_features.append(final_features)

        X = np.concatenate(list_of_features)
        del list_of_features
        y = np.concatenate(list_of_labels)
        del list_of_labels

        return X,y



 #---------------------------------------------------------------------------------------------------------------------
 
    @staticmethod
    def gyro_integration(ts, gyro, init_q):
        """
        Integrate gyro into orientation.
        https://www.lucidar.me/en/quaternions/quaternion-and-gyroscope/
        """
        output_q = np.zeros((gyro.shape[0], 4))
        output_q[0] = init_q
        dts = ts[1:] - ts[:-1]
        for i in range(1, gyro.shape[0]):
            output_q[i] = output_q[i - 1] + data_loader.angular_velocity_to_quaternion_derivative(output_q[i - 1], gyro[i - 1]) * dts[
                i - 1]
            output_q[i] /= np.linalg.norm(output_q[i])
        return output_q


    @staticmethod
    def select_orientation_source(data_path, max_ori_error=20.0, grv_only=True, use_ekf=True):
        """
        Select orientation from one of gyro integration, game rotation vector or EKF orientation.

        Args:
            data_path: path to the compiled data. It should contain "data.hdf5" and "info.json".
            max_ori_error: maximum allow alignment error.
            grv_only: When set to True, only game rotation vector will be used.
                    When set to False:
                        * If game rotation vector's alignment error is smaller than "max_ori_error", use it.
                        * Otherwise, the orientation will be whichever gives lowest alignment error.
                    To force using the best of all sources, set "grv_only" to False and "max_ori_error" to -1.
                    To force using game rotation vector, set "max_ori_error" to any number greater than 360.
        Returns:
            source_name: a string. One of 'gyro_integration', 'game_rv' and 'ekf'.
            ori: the selected orientation.
            ori_error: the end-alignment error of selected orientation.
        """
        ori_names = ['gyro_integration', 'game_rv']
        ori_sources = [None, None, None]

        with open(os.path.join(data_path, 'info.json')) as f:
            info = json.load(f)
            ori_errors = np.array(
                [info['gyro_integration_error'], info['grv_ori_error'], info['ekf_ori_error']])
            init_gyro_bias = np.array(info['imu_init_gyro_bias'])

        with h5py.File(os.path.join(data_path, 'data.hdf5')) as f:
            ori_sources[1] = np.copy(f['synced/game_rv'])
            if grv_only or ori_errors[1] < max_ori_error:
                min_id = 1
            else:
                if use_ekf:
                    ori_names.append('ekf')
                    ori_sources[2] = np.copy(f['pose/ekf_ori'])
                min_id = np.argmin(ori_errors[:len(ori_names)])
                # Only do gyro integration when necessary.
                if min_id == 0:
                    ts = f['synced/time']
                    gyro = f['synced/gyro_uncalib'] - init_gyro_bias
                    ori_sources[0] = data_loader.gyro_integration(ts, gyro, ori_sources[1][0])

        return ori_names[min_id], ori_sources[min_id], ori_errors[min_id]


    @staticmethod
    def sliding_window_label(input_array, step_size, window_size):
        """
        Function that takes the 0 and the nth element of the list amd computes a difference between them in a sliding window

        Args:
            input_array: list of labels, dimension = [n,2]
            step_size: stepsize of the sliding window (default = 10)
            window_size: size of the sliding window, i.e. how many elements are in the window (200 Hz = 200 elements)

        Returns:
            output_array: 2D array with labels, dimension = [n,2]
        """
        default_window_size = window_size
        output_array = []
        for i in range(0, len(input_array) - window_size, step_size):
            
            # check for NaN values in the window
            while (np.isnan(input_array[i]).any()) or np.isnan(input_array[i+window_size]).any():
                if i >= len(input_array)-window_size-1:
                    break
                else:
                    i += 1

            output_array.append(input_array[i+window_size] - input_array[i])
            window_size = default_window_size
        return np.array(output_array)[:,:2]


    @staticmethod
    def sliding_window_features(input_array, step_size, window_size):
        """
        Function takes n x m elements of given array in a sliding window and appends them to a new array

        Args:
            input_array: list of labels, dimension = [n,6]
            step_size: stepsize of the sliding window (default = 10)
            window_size: size of the sliding window, i.e. how many elements are in the window (200 Hz = 200 elements)

        Returns:
            output_array: 3D array with features, dimension = [n,window_size, 6]
        """
        default_window_size = window_size
        output_array = []
        for i in range(0, len(input_array) - window_size, step_size):
            while (np.isnan(input_array[i]).any() or np.isnan(input_array[i+window_size]).any()):
                if i >= len(input_array)-window_size-1:
                    break
                else:
                    i += 1
            output_array.append(input_array[i:i+window_size])
            window_size = default_window_size
        return np.array(output_array)  


    @staticmethod
    def slider(features, labels, step_size, window_size):
        feature_output = []
        label_output = []
        for i in range(0, len(features) - window_size, step_size):
            while (np.isnan(labels[i]).any() or np.isnan(labels[i+window_size]).any()):
                if i >= len(labels)-window_size-1:
                    break
                else:
                    i += 1
            label_output.append(labels[i+window_size] - labels[i])
            feature_output.append(features[i:i+window_size])

        return_labels = np.array(label_output)[:,:2]
        return_features = np.array(feature_output)

        return return_features, return_labels


# convert roll pitch yaw to quaternion
def quat_from_rpy(x, y, z):
    #q = quaternion.quaternion(np.cos(yaw/2), np.sin(yaw/2)*np.cos(roll/2), np.sin(yaw/2)*np.sin(roll/2), np.sin(yaw/2)*np.sin(pitch/2))
    #return q
    rot = Rotation.from_euler('xyz', [x,y,z], degrees=False)
    rot_quat = rot.as_quat()
    q = quaternion.from_float_array(rot_quat)
    return q



# loop to convert roll pitch yaw to an array with quaternions
def quat_from_rpy_loop(roll, pitch, yaw):
    """
    Function to convert roll pitch yaw to an array with quaternions.
    
    Args:
        roll: roll angle in radians.
        pitch: pitch angle in radians.
        yaw: yaw angle in radians.
        
        Returns:
            quat: array with quaternions."""
    q_list = []
    for i in range(len(roll)):
        q = quat_from_rpy(roll[i], pitch[i], yaw[i])
        q_list.append(q)
    return np.array(q_list)
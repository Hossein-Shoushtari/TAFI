import numpy as np
#from scipy import integrate
from ahrs.filters import Madgwick


def compute_stride_length(feat_4_std_acc,feat_6_max_acc,feat_8_min_acc):
    """
    compute the stride length

    Parameters
    ----------
    step std, max, mins: stride parameters

    Returns
    -------
    stride_lengths: stride length
    """
    K = 0.4
    K_max = 0.8
    K_min = 0.4
    para_a0 = 0.21468084
    para_a1 = 0.09154517
    para_a2 = 0.02301998
    # calculate parameters acceleration magnitude variance
    k_real= np.max([(para_a0 + para_a1 + para_a2 * feat_4_std_acc), K_min])
    k_real = np.min([k_real, K_max]) * (K / K_min)

    # calculate every stride length by parameters and max and min data of acceleration magnitude, 1 sec
    stride_length = np.max([(feat_6_max_acc - feat_8_min_acc),1]) ** (1 / 4) * k_real

    return stride_length

def count_peaks_troughs(x):
    # Find the differences between adjacent elements
    dx = np.diff(x)

    # Find the sign of the differences
    sign = np.sign(dx)

    # Find the indices of the sign changes
    indices = np.where(np.diff(sign))[0] + 1

    # Find the values at those indices
    values = x[indices]

    # Count the number of peaks and troughs
    num_peaks = np.sum(sign[:-1] == 1)
    num_troughs = np.sum(sign[:-1] == -1)

    return num_peaks, num_troughs

def feature_calculator(a):
    #a include N rows of measuremnt with columns gyr_x, gyr_y, gyr_z and gyr_x, gyr_y, gyr_z.

    # Calculate the magnitude of the gyr vector between column 0 till 2
    mag_gyr = np.sqrt(a[:, 0]**2 + a[:, 1] **2 + a[:, 2] **2)
    # Calculate the magnitude of the acc vector between column 3 till 5
    mag_acc = np.sqrt(a[:, 3]**2 + a[:, 4] **2 + a[:, 5] **2)

    #orientation free physical context features calculation:
    # Calculate the mean of the array
    feat_1_mean_gyr = np.mean(mag_gyr)
    feat_2_mean_acc = np.mean(mag_acc)

    # Calculate the standard deviation of the array
    feat_3_std_gyr = np.std(mag_gyr)
    feat_4_std_acc = np.std(mag_acc)

    # Calculate the max of the array
    feat_5_max_gyr = np.max(mag_gyr)
    feat_6_max_acc = np.max(mag_acc)

    # Calculate the min of the array
    feat_7_min_gyr = np.min(mag_gyr)
    feat_8_min_acc = np.min(mag_acc)

    # Calculate the energy of the array
    feat_9_energy_gyr = np.sum(mag_gyr**2)
    feat_10_energy_acc = np.sum(mag_acc**2)

    # Calculate the sum of the array
    feat_11_sum_gyr = np.sum(mag_gyr)
    feat_12_sum_acc = np.sum(mag_acc)

    # Calculate the amplitude of the array
    feat_13_amplitude_gyr = np.max(np.abs(mag_gyr))
    feat_14_amplitude_acc = np.max(np.abs(mag_acc))

    # Calculate the range of the array
    feat_15_range_gyr = np.max(mag_gyr) - np.min(mag_gyr)
    feat_16_range_acc = np.max(mag_acc) - np.min(mag_acc)

    # Calculate the correlation between 'mag_gyr' and 'mag_acc'
    feat_17_correlation = np.corrcoef(mag_gyr, mag_acc)[0, 1]

    # Count the number of peaks and troughs of the array
    feat_18_num_peaks_gyr, feat_19_num_troughs_gyr = count_peaks_troughs(mag_gyr)
    feat_20_num_peaks_acc, feat_21_num_troughs_acc = count_peaks_troughs(mag_acc)

    # calculate the step length
    feat_17_step_length = compute_stride_length(feat_4_std_acc,feat_6_max_acc,feat_8_min_acc)

    # Calculate the simpson integrate of the vectors 'mag_gyr' and 'mag_acc'
    #feat_22_integration_gyr = integrate.simpson(np.ones_like(mag_gyr),mag_gyr)
    #feat_23_integration_acc = integrate.simpson(np.ones_like(mag_acc),mag_acc)

    # calculate the Madgwick Algorithm as feature
    madgwick_filter = Madgwick(beta=1)
    # Update the filter with gyroscope and accelerometer data of the last measurement in time windew
    q_data = np.array([1., 0., 0., 0.])
    last_gyr = np.array([a[-1, 0], a[-1, 1], a[-1, 2]])
    last_acc = np.array([a[-1, 3], a[-1, 4], a[-1, 5]])
    feat_24_quaternion = madgwick_filter.updateIMU(q_data,last_gyr,last_acc)

    #output = [feat_20_num_peaks_acc]
    output = [feat_1_mean_gyr, feat_2_mean_acc, feat_3_std_gyr, feat_4_std_acc, feat_5_max_gyr, feat_6_max_acc,
              feat_7_min_gyr, feat_8_min_acc, feat_9_energy_gyr, feat_10_energy_acc, feat_11_sum_gyr, feat_12_sum_acc,
              feat_13_amplitude_gyr, feat_14_amplitude_acc, feat_15_range_gyr, feat_16_range_acc, feat_17_step_length,
              feat_18_num_peaks_gyr,
              feat_19_num_troughs_gyr, feat_20_num_peaks_acc,feat_21_num_troughs_acc,
              np.float32(feat_24_quaternion[0]),
              np.float32(feat_24_quaternion[1]), np.float32(feat_24_quaternion[2]), np.float32(feat_24_quaternion[3])]
    # feat_13_amplitude_gyr, feat_14_amplitude_acc, feat_15_range_gyr, feat_16_range_acc, feat_17_correlation,

    # has_feature_nan = np.any(np.isnan(output))
    # if has_feature_nan:
    #     # Find the indices of NaN values
    #     nan_indices = np.isnan(output)
    #     # Delete the elements with NaN values
    #     output = output[~nan_indices]

    return np.array(output, dtype=np.float32)

#
# N = 10
# #a: gyr_x gyr_x gyr_x acc_x acc_x acc_x with N rows
# a = np.random.rand(N, 6)
# print(a.shape)
# b = feature_calculator(a)
# print(b)
# print
#
# #preparation for federated learning: calculate the frequency of the sensors and handel the deep learning w.r.t freq
#
#



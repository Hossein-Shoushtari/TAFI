import numpy as np
import matplotlib.pyplot as plt
from metric import compute_ate_rte2, errors_sqrt, CDF_params, distance_time_below_threshold
import glob
import os
import pandas as pd

def read_the_file(path):
    df = pd.read_csv(path)

    df['prediction_x'] = df['Prediction'].apply(lambda x: float(x.strip('[]').split()[0])).cumsum()
    df['prediction_y'] = df['Prediction'].apply(lambda x: float(x.strip('[]').split()[1])).cumsum()

    df['target_x'] = df['Target'].apply(lambda x: float(x.strip('[]').split()[0])).cumsum()
    df['target_y'] = df['Target'].apply(lambda x: float(x.strip('[]').split()[1])).cumsum()

    # Remove the original "Prediction" column if needed
    df = df.drop('Prediction', axis=1)
    df = df.drop('Target', axis=1)

    numpy_array = df.values

    prediction = numpy_array[:,0:2]
    target = numpy_array[:,2:4]
    return prediction, target


## PLOT ======================================================================================================================
def plot_trajCDF(path, target: list, prediction: list, _id: int, errors: list, ate: float, rte: float,
                 time_below_1m: float, distance_below_1m: float, CEP50: float, CEP90: float) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    # * left plot
    axs[0].set_xlabel("Abs. Error [m]")
    axs[0].set_ylabel("Cummulative Probability [0...1]")
    axs[0].plot(
        np.sort(errors),
        np.linspace(0, 1, errors.shape[0]),
        color="royalblue",
        linewidth=1.2,
    )
    # 50/90 lines
    axs[0].axhline(0.9, color="#535353", linestyle="--", linewidth=0.8)
    axs[0].axhline(0.5, color="#535353", linestyle="--", linewidth=0.8)
    # 50/90 text boxes
    axs[0].text(
        max(errors) - (max(errors) - min(errors)) * 0.08,
        0.5,  # position in y
        f"CEP50: {CEP50:.2f}",  # text
        color="#535353",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="white"),
        ha="center",
        va="center",
    )
    axs[0].text(
        max(errors) - (max(errors) - min(errors)) * 0.08,
        0.9,  # position in y
        f"CEP90: {CEP90:.2f}",  # text
        color="#535353",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="white"),
        ha="center",
        va="center",
    )
    # textbox containing ATE and RTE on the left plot at the right bottom corner
    axs[0].text(
        max(errors) - (max(errors) - min(errors)) * 0.18,
        0.05,
        f"ATE: {ate:.2f} m\nRTE: {rte:.2f} m\nTime below 1 m: {time_below_1m:.2f} s\nDistance below 1 m: {distance_below_1m:.2f} m",
        color="#535353",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.2"),
        ha="center",
        va="center",
    )
    axs[0].grid(color="gray", linestyle="-", linewidth=0.4)

    # * right plot
    axs[1].set_xlabel("X [m]")
    axs[1].set_ylabel("Y [m]")
    axs[1].plot(prediction[:, 0], prediction[:, 1], color="orange", linewidth=1.2)
    axs[1].plot(target[:, 0], target[:, 1], color="green", linewidth=1.2)
    axs[1].legend(["Prediction", "Target"], fontsize=10)
    axs[1].grid(color="gray", linestyle="-", linewidth=0.4)

    # save plot

    fig.savefig(f"{path}/traj{_id}.png", dpi=300)


def plot_final(path,all_errors: list, mean_ATE: float, mean_RTE: float, mean_time_below_1m: float,
               mean_distance_below_1m: float, CEP50: float, CEP90: float) -> None:
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    # * left plot
    axs.set_xlabel("Abs. Error [m]")
    axs.set_ylabel("Cummulative Probability [0...1]")
    axs.plot(
        np.sort(all_errors),
        np.linspace(0, 1, all_errors.shape[0]),
        color="royalblue",
        linewidth=1.2,
    )
    # 50/90 lines
    axs.axhline(0.9, color="#535353", linestyle="--", linewidth=0.8)
    axs.axhline(0.5, color="#535353", linestyle="--", linewidth=0.8)
    # 50/90 text boxes
    axs.text(
        max(all_errors) - (max(all_errors) - min(all_errors)) * 0.04,
        0.5,  # position in y
        f"CEP50: {CEP50:.2f}",  # text
        color="#535353",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="white"),
        ha="center",
        va="center",
    )
    axs.text(
        max(all_errors) - (max(all_errors) - min(all_errors)) * 0.04,
        0.9,  # position in y
        f"CEP90: {CEP90:.2f}",  # text
        color="#535353",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="white"),
        ha="center",
        va="center",
    )
    # textbox containing ATE and RTE on the left plot at the right bottom corner
    axs.text(
        max(all_errors) - (max(all_errors) - min(all_errors)) * 0.17,
        0.13,
        f"mean ATE: {mean_ATE:.2f} m\nmean RTE: {mean_RTE:.2f} m\nmean time below 1 m: {mean_time_below_1m:.2f} s\nmean distance below 1 m: {mean_distance_below_1m:.2f} m",
        color="#535353",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.2"),
        ha="center",
        va="center",
    )
    axs.grid(color="gray", linestyle="-", linewidth=0.4)


    # show plot
    fig.show()

    # save plot
    fig.savefig(f"{path}/dataset.png", dpi=300)

#----------main

path = "data/tafi/ridi"
csv_files = glob.glob(os.path.join(path, '*.csv'))
# Count the number of CSV files
num_csv_files = len(csv_files)

all_errors = []
ates = []
rtes = []
times_below_1m = []
distances_below_1m = []
for i in range(0,num_csv_files):
    path_csv = csv_files[i]
    prediction, target = read_the_file(path_csv)
    errors = errors_sqrt(prediction,target)
    all_errors.extend(errors)
    CEP50, CEP90 = CDF_params(errors)
    ate, rte = compute_ate_rte2(prediction, target)
    ates.append(ate)
    rtes.append(rte)
    time_below_1m,distance_below_1m = distance_time_below_threshold(errors,prediction,1)
    times_below_1m.append(time_below_1m)
    distances_below_1m.append(distance_below_1m)

    # ? PLOT ===================================================================================
    plot_trajCDF(path, target, prediction, i, errors, ate, rte, time_below_1m, distance_below_1m, CEP50, CEP90)

    # ? STORE DATA =============================================================================

    with open(f"{path}/all_erorrs.txt", "a") as f:
        for k in range(len(target)):
            f.write(
                f"{i + 1},{prediction[k, 0]},{prediction[k, 1]},{target[k, 0]},{target[k, 1]},{errors[k]}\n"
            )

    # print progress
    print(f"{i + 1}/{num_csv_files} metrics estimated.")

    # ? PLOT FINAL PLOT ============================================================================
all_errors = np.array(all_errors)
mean_ATE = np.mean(ates)
mean_RTE = np.mean(rtes)
mean_time_below_1m = np.mean(times_below_1m)
mean_distance_below_1m = np.mean(distances_below_1m)

CEP50_index = np.abs(np.linspace(0, 1, all_errors.shape[0]) - 0.5).argmin()
CEP90_index = np.abs(np.linspace(0, 1, all_errors.shape[0]) - 0.9).argmin()
CEP50 = np.sort(all_errors)[CEP50_index]
CEP90 = np.sort(all_errors)[CEP90_index]

plot_final(path,all_errors, mean_ATE, mean_RTE, mean_time_below_1m, mean_distance_below_1m, CEP50, CEP90)






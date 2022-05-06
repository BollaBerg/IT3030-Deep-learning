from pathlib import Path

import matplotlib.pyplot as plt

from src.helpers.path import ROOT_PATH

def plot_losses(loss_list: list, savepath: Path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set_title("Hyperparameter differences on validation loss")
    ax.set_yscale("log")
    ax.set_ylabel("log(loss), MSE")
    ax.set_xlabel("Epoch")

    colormap = plt.get_cmap("Set1").colors + plt.get_cmap("Accent").colors

    for i, (model_name, loss_location) in enumerate(loss_list):
        with open(loss_location, "r") as file:
            losses = file.readlines()
            losses_float = [float(loss) for loss in losses]

            ax.plot(losses_float, label=model_name, color=colormap[i])
    
    ax.legend()
    fig.savefig(savepath)


if __name__ == "__main__":
    task_1_loss_list = [
        ("All options off", "models/losses/1_all_off.losses"),
        ("Time columns on", "models/losses/2_time_on.losses"),
        ("Past targets on", "models/losses/3_hanging_y.losses"),
        ("All extras on", "models/losses/4_all_on.losses"),
        ("Time of day and week", "models/losses/5_day_week.losses"),
        ("Time of day, year + past targets", "models/losses/6_day_year_hanging.losses"),
        ("Time of day, year + past targets (30 steps)", "models/losses/7_day_year_targets_more_memory.losses"),
        ("Randomness in last_y", "models/losses/8_randomness_last_y.losses"),
        ("Time of day, year + past targets (dropout)", "models/losses/10_6_with_dropout.losses"),
        ("Time of day, year + past targets (only 1 LSTMcell)", "models/losses/11_6_with_smaller_depth.losses"),
        ("Time of day, year + past targets (only 1 LSTMcell + dropout)", "models/losses/12_6_with_smaller_depth_with_dropout.losses"),
    ]
    plot_losses(task_1_loss_list, ROOT_PATH / "plots/model_differences_task_1.png")

    task_2_loss_list = [
        ("All options off", "models/losses/13_task_2_all_off.losses"),
        ("Time of day, year + past targets", "models/losses/9_task_2_from_6.losses"),
        ("Time of day, year + past targets (with dropout)", "models/losses/14_task_2_from_6_with_dropout.losses"),
    ]
    plot_losses(task_2_loss_list, ROOT_PATH / "plots/model_differences_task_2.png")


import matplotlib.pyplot as plt

from src.helpers.path import ROOT_PATH

def plot_losses(loss_list: list):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.set_title("Hyperparameter differences on validation loss")
    ax.set_yscale("log")
    ax.set_ylabel("log(loss), MSE")
    ax.set_xlabel("Epoch")

    colormap = plt.get_cmap("Paired").colors

    for i, (model_name, loss_location) in enumerate(loss_list):
        with open(loss_location, "r") as file:
            losses = file.readlines()
            losses_float = [float(loss) for loss in losses]

            ax.plot(losses_float, label=model_name, color=colormap[i])
    
    ax.legend()
    fig.savefig(ROOT_PATH / "plots/model_differences.png")


if __name__ == "__main__":
    loss_list = [
        ("All options off", "models/losses/1_all_off.losses"),
        ("Time columns on", "models/losses/2_time_on.losses"),
    ]
    plot_losses(loss_list)

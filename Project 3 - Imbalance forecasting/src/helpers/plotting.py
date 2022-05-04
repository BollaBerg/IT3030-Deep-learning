import pathlib

import matplotlib.pyplot as plt
import torch

from src.helpers.path import ROOT_PATH


def plot_loss_data(losses: list,
                    ax: plt.Axes,
                    title: str = "Losses per epoch",
                    log_y: bool = False):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.plot(losses)
    if log_y:
        ax.set_yscale("log")


def plot_validation_prediction(model,
                                dataloader,
                                epoch: int,
                                postprocess_target = lambda x: x,
                                ax: plt.Axes = None,
                                base_save_path: pathlib.Path = None
    ):
    if ax is None:
        create_new = True
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))

        if base_save_path is None:
            savepath = ROOT_PATH / f"training/plots/LSTM_{epoch}.png"
        else:
            savepath = pathlib.Path(base_save_path) / f"plots/LSTM_{epoch}.png"
    else:
        create_new = False

    ax.set_title(f"Epoch {epoch}, single day prediction")

    with torch.no_grad():
        # Assumes that dataloader has batchsize = full length, so this is
        # simply a way to get the full dataset
        for inputs, targets in dataloader:
            outputs = model(inputs)

            post_outputs = postprocess_target(outputs).detach().numpy().flatten()
            post_targets = postprocess_target(targets).detach().numpy().flatten()

            ax.plot(post_outputs, label="Model outputs")
            ax.plot(post_targets, label="Targets")
            ax.fill_between(
                range(len(post_outputs)), post_outputs, post_targets,
                color="black", alpha=0.1
            )
            ax.set_ylabel("Predicted output")

            # ax_diff = ax.twinx()
            # difference = post_outputs - post_targets
            # ax_diff.plot(difference.detach().numpy(), label="Difference", color="cyan")
            # ax_diff.set_ylabel("Predicted output - targets", color="cyan")
            # ax_diff.tick_params(axis="y", labelcolor="cyan")

    ax.set_xticks([])
    ax.legend()
    # ax_diff.legend()
    
    if create_new:
        fig.savefig(savepath)
        plt.close(fig)


def plot_future_predictions(
            predictions: torch.Tensor,
            targets: torch.Tensor,
            title: str = "",
            postprocess_target = lambda x: x,
            ax: plt.Axes = None,
            savepath: str = None
            ):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    else:
        fig = None

    post_predictions = postprocess_target(predictions).detach().numpy().flatten()
    post_targets = postprocess_target(targets).detach().numpy().flatten()

    ax.set_title(title)
    ax.plot(post_predictions, label="Model outputs")
    ax.plot(post_targets, label="Targets")
    ax.fill_between(
        range(len(post_targets)), post_predictions, post_targets,
        color="black", alpha=0.1
    )
    ax.set_ylabel("Predicted output")
    ax.set_xticks([])
    ax.legend()

    if savepath is not None:
        if fig is None:
            raise ValueError("Cannot save existing ax")
        else:
            fig.savefig(savepath)

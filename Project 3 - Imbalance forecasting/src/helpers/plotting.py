import pathlib
import random

import matplotlib.pyplot as plt
import torch

from src.helpers.path import ROOT_PATH


def plot_loss_data(losses: list,
                    ax: plt.Axes,
                    title: str = "Losses per epoch",
                    label: str = "",
                    log_y: bool = False):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.plot(losses, label=label)
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
            ax: plt.Axes = None,
            savepath: str = None,
            loss_str: str = "",
            ):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), layout="tight")
    else:
        fig = None

    post_predictions = predictions.detach().numpy().flatten()
    post_targets = targets.detach().numpy().flatten()


    ax.set_title(title, fontsize=18)
    ax.text(x=0.5, y=0.90, s=loss_str, fontsize=12, ha="center", transform=ax.transAxes)
    ax.plot(
        post_predictions,
        label="Model outputs",
        color=plt.get_cmap("Set1").colors[1]
    )
    ax.plot(
        post_targets,
        label="Targets",
        color=plt.get_cmap("Set2").colors[0]
    )
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
            print(f"Predictions saved to {savepath}")
            plt.close(fig)


def plot_zoomed_future_predictions(
            predictions: torch.Tensor,
            targets: torch.Tensor,
            title: str = "",
            ax: plt.Axes = None,
            savepath: str = None,
            start_index: int = None,
            width_of_zoom: int = 60 * 24 * 2
            ):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10), layout="tight")
    else:
        fig = None
    
    if start_index is None:
        start_index = random.randrange(0, len(predictions) - width_of_zoom)

    plot_future_predictions(
        predictions[start_index: start_index + width_of_zoom],
        targets[start_index: start_index + width_of_zoom],
        title=title + f", starting at timestep {start_index}",
        ax=ax,
    )
    
    if savepath is not None:
        if fig is None:
            raise ValueError("Cannot save existing ax")
        else:
            fig.savefig(savepath)
            print(f"Zoomed predictions saved to {savepath}")
            plt.close(fig)


def plot_predictions_from_single_starts(
            predictions: list[tuple[torch.Tensor, torch.Tensor]],
            savepath: str,
            start_indices: list = None,
            previous_steps: int = 30,
        ):
    past_color = plt.get_cmap("Paired").colors[1]
    target_color = plt.get_cmap("Paired").colors[0]
    pred_color = plt.get_cmap("Paired").colors[3]

    prediction_steps: int = len(predictions)

    if start_indices is None:
        # Default to 4 random start-positions, between 0 and the length of the
        # 2-hour predictions, with padding enough to make predictions 24 steps
        # into the future
        start_indices = [
            random.randrange(previous_steps + 1, len(predictions[-1][0]) - prediction_steps)
            for _ in range(8)
        ]
    fig, axes = plt.subplots(
        nrows=int(len(start_indices) / 2),
        ncols=2,
        figsize=(10, 20),
        layout="tight"
    )
    ax = axes.flat
    for i, start_index in enumerate(start_indices):
        past_data = predictions[0][1][start_index - previous_steps: start_index + 1]
        preds = [predictions[i][0][start_index + i].item() for i in range(prediction_steps)]
        targets = predictions[0][1][start_index: start_index + prediction_steps]

        ax[i].plot(past_data, color=past_color, label="Actual data before predictions")
        ax[i].plot(
            range(previous_steps, previous_steps + prediction_steps),
            preds,
            color=pred_color,
            label="Predictions"
        )
        ax[i].plot(
            range(previous_steps, previous_steps + prediction_steps),
            targets,
            color=target_color,
            label="Target",
            linestyle="--"
        )

        ax[i].set_title(f"Predictions starting at timestep {start_index}")
        ax[i].legend()
        ax[i].set_ylabel("Predicted output")
        ax[i].set_xticks([])

    fig.savefig(savepath)
    print(f"Single-start predictions saved to {savepath}")
    plt.close(fig)


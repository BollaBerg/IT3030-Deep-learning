from pathlib import Path

from torch.nn import MSELoss

from src.helpers.core import format_steps
from src.helpers.path import ROOT_PATH
from src.helpers.plotting import (
    plot_future_predictions, plot_zoomed_future_predictions,
    plot_predictions_from_single_starts
)
from src.future_prediction import predict_future_from_paths


def demo(config_path: Path, model_path: Path, data_path: Path, task_2: bool = False):
    future_steps = int(120 / 5)      # 2 hours = 120 min / 5 min timesteps
    loss_fn = MSELoss(reduction="mean")

    if task_2:
        model_name = "TASK_2"
    else:
        model_name = "LSTM"

    predictions = predict_future_from_paths(
        config_path, model_path, data_path, future_steps
    )

    for i, future_step in enumerate(predictions):
        loss = loss_fn(future_step[0], future_step[1]).item()
        plot_future_predictions(
            future_step[0], future_step[1],
            title=f"Predictions, {format_steps(i)} into the future",
            savepath = ROOT_PATH / f"plots/{model_name}_future_{i}.png",
            loss_str=f"Loss: {loss}"
        )
    
    # Plot a zoomed window of 2-hour predictions
    for zoom_i in range(5):
        plot_zoomed_future_predictions(
            predictions[-1][0], predictions[-1][1],
            title="Zoomed predictions, 2 hours into the future",
            savepath=ROOT_PATH / f"plots/{model_name}_zoomed_{zoom_i}.png"
        )

    # Plot four plots of the previous target, then predictions over two hours
    plot_predictions_from_single_starts(
        predictions,
        savepath=ROOT_PATH / f"plots/{model_name}_over_time.png"
    )

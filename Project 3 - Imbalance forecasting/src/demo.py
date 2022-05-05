from pathlib import Path

from torch.nn import MSELoss

from src.helpers.path import ROOT_PATH
from src.helpers.plotting import (
    plot_future_predictions, plot_zoomed_future_predictions,
    plot_predictions_from_single_starts
)
from src.future_prediction import predict_future_from_paths


def _format_steps(step: int) -> str:
    minutes = step * 5
    hours = minutes // 60
    minutes -= 60 * hours
    
    if hours == 1:
        return f"{hours} hour, {minutes} minutes"
    elif hours > 1:
        return f"{hours} hours, {minutes} minutes"
    else:
        return f"{minutes} minutes"


def demo(config_path: Path, model_path: Path, data_path: Path):
    future_steps = int(120 / 5)      # 2 hours = 120 min / 5 min timesteps
    loss_fn = MSELoss(reduction="mean")

    predictions = predict_future_from_paths(
        config_path, model_path, data_path, future_steps, loss_fn
    )

    for i, future_step in enumerate(predictions):
        loss = loss_fn(future_step[0], future_step[1]).item()
        plot_future_predictions(
            future_step[0], future_step[1],
            title=f"Predictions, {_format_steps(i)} into the future",
            savepath = ROOT_PATH / f"plots/LSTM_future_{i}.png",
            loss_str=f"Loss: {loss}"
        )
    
    # Plot a zoomed window of 2-hour predictions
    plot_zoomed_future_predictions(
        predictions[-1][0], predictions[-1][1],
        title="Zoomed predictions, 2 hours into the future",
        savepath=ROOT_PATH / "plots/LSTM_zoomed.png"
    )

    # Plot four plots of the previous target, then predictions over two hours
    plot_predictions_from_single_starts(
        predictions,
        savepath=ROOT_PATH / "plots/LSTM_over_time.png"
    )

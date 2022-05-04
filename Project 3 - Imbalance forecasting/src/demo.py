from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from src.helpers.config import read_config
from src.helpers.dataset import FutureDataset
from src.helpers.path import DATA_PATH, ROOT_PATH
from src.helpers.plotting import plot_future_predictions
from src.future_prediction import predict_into_future
from src.lstm import LSTM
from src.preprocessing import Preprocesser, split_data, pd_to_tensor


def _format_steps(step: int) -> str:
    minutes = step * 5
    hours = minutes // 60
    minutes -= 60 * hours
    
    if hours > 0:
        return f"{hours} hours, {minutes} minutes"
    else:
        return f"{minutes} minutes"


def demo(config_path: Path, model_path: Path, data_path: Path):
    future_steps = 120 / 5      # 2 hours = 120 min / 5 min timesteps
    config = read_config(config_path)
    input_size = (
        8   # Standard, from the actual input data
        + (2 if config.data.time_of_day else 0)    # time_of_day has two cols
        + (2 if config.data.time_of_week else 0)   # time_of_week has two cols
        + (2 if config.data.time_of_year else 0)   # time_of_year has two cols
        + (1 if config.data.last_day_y else 0)     # last_day_y has one col
        + (1 if config.data.two_last_day_y else 0) # two_last_day_y has one col
    )
    model = LSTM(
        input_size = input_size,
        lstm_depth=config.model.lstm_depth,
        hidden_layers=config.model.hidden_layers
    )
    model.load_model(model_path)

    data = pd.read_csv(data_path, parse_dates=["start_time"])

    # Read training data only to fit preprocesser
    _training_data = pd.read_csv(DATA_PATH / "train.csv", parse_dates=["start_time"])
    preprocesser = Preprocesser(
        time_of_day=config.data.time_of_day,
        time_of_week=config.data.time_of_week,
        time_of_year=config.data.time_of_year,
        last_day_y=config.data.last_day_y,
        two_last_day_y=config.data.two_last_day_y
    )
    preprocesser.fit(_training_data)

    processed_data = preprocesser.transform(data)
    inputs_df, target_df = split_data(processed_data)
    inputs = pd_to_tensor(inputs_df)
    targets = pd_to_tensor(target_df)

    dataset = FutureDataset(inputs, targets,
        past_window=config.data.sequence_length,
        future_window=future_steps
    )
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    predictions = predict_into_future(
        model, dataloader,
        timesteps_into_future=future_steps,
        reverse_target=preprocesser.reverse_y
    )

    for i, future_step in enumerate(predictions):
        plot_future_predictions(
            future_step[0], future_step[1],
            title=f"Predictions, {_format_steps(i)} into the future",
            savepath = ROOT_PATH / f"plots/LSTM_future_{i}.png"
        )

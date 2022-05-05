from pathlib import Path

import pandas as pd
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from src.helpers.config import read_config
from src.helpers.dataset import FutureDataset
from src.helpers.path import DATA_PATH
from src.lstm import LSTM
from src.preprocessing import Preprocesser, split_data, pd_to_tensor

def predict_into_future(model: LSTM,
                        dataloader: DataLoader,
                        timesteps_into_future: int,
                        reverse_target = lambda x: x
                    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Predict N timesteps into the future

    Args:
        model (LSTM): Trained LSTM model
        dataloader (DataLoader): DataLoader built on FutureDataset
        timesteps_into_future (int): N, number of timesteps into the future 
            which should be predicted

    Returns:
        list[tuple[torch.Tensor, torch.Tensor]]: List of tuples of Tensors.
            Each element of the list is a tuple of tensors - (outputs, targets).
            I.e. returns a list with structure
                [
                    (output_0, target_0),
                    (output_1, target_1),
                    ...
                ]
    """
    if not isinstance(dataloader.dataset, FutureDataset):
        raise ValueError("Dataloader must be based on FutureDataset")
    
    past_window = dataloader.dataset.past_window
    output = []

    with torch.no_grad():
        # Needed to get full data from dataloader
        for inputs, targets in dataloader:

            # Iterate through future timesteps
            for future_timestep in range(timesteps_into_future + 1):
                # For each timestep, get a past_window length sequence of data,
                # starting at future_timestep (i.e. start at 0, then walk up) 
                timestep_in = inputs[:, future_timestep:future_timestep + past_window, :]

                # Do a standard prediction
                timestep_prediction = model(timestep_in)
                timestep_target = targets[:, future_timestep, :]

                # Add prediction and target to output list
                output.append(
                    (
                        reverse_target(timestep_prediction),
                        reverse_target(timestep_target)
                    )
                )

                # Replace the next input's "last_y" with the recently predicted
                # value. This means we use the predicted value, rather than the
                # input value, as input for our next predictions
                # This answers the assignment, and is also needed for multi-step
                # predictions
                if future_timestep != timesteps_into_future:
                    # We do not update for the last step, as we have no next-
                    # input to replace "previous_y" at
                    inputs[:, future_timestep + past_window, -1] = timestep_prediction.flatten()
    
    return output


def predict_future_from_paths(
        config_path: Path,
        model_path: Path,
        data_path: Path,
        future_steps: int = int(120 / 5),
    ):
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

    return predict_into_future(
        model, dataloader,
        timesteps_into_future=future_steps,
        reverse_target=preprocesser.reverse_y
    )


if __name__ == "__main__":
    model = LSTM(3, 2, 2)
    inputs = torch.tensor([list(range(9)), list(range(10, 19)), list(range(20, 29))], dtype=float).T
    targets = torch.tensor(list(range(90, 99)), dtype=float).reshape((-1, 1))
    dataset = FutureDataset(inputs, targets, past_window=2, future_window=2)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    print(predict_into_future(model, dataloader, 2))
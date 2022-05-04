import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import MSELoss
from torch.optim import Adam, LBFGS
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from src.helpers.config import read_config, Config
from src.helpers.dataset import TimeSeriesDataset
from src.helpers.path import DATA_PATH, ROOT_PATH
from src.helpers.plotting import plot_loss_data, plot_validation_prediction
from src.lstm import LSTM
from src.preprocessing import Preprocesser, split_data, pd_to_tensor

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def _train_epoch(model, dataloader, optimizer, loss_fn):
    for inputs, targets in dataloader:
        # if LBFGS
        def closure():
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            return loss
        optimizer.step(closure)

        # if ADAM
        # optimizer.zero_grad()
        # output = model(inputs)
        # loss = loss_fn(output, targets)
        # loss.backward()
        # optimizer.step()


def _get_validation_loss(model, dataloader, loss_fn):
    losses = []
    with torch.no_grad():
        for val_input, val_target in dataloader:
            val_output = model(val_input)
            loss = loss_fn(val_output, val_target)

            losses.append(loss)
    
    return sum(losses) / len(losses)


def _save_losses(losses, path):
    str_losses = "\n".join([str(loss.item()) for loss in losses])
    with open(path, "w") as file:
        file.write(str_losses)


def train_model(config: Config,
                base_save_path: pathlib.Path = ROOT_PATH / "training"):
    base_save_path = pathlib.Path(base_save_path)

    # Setup model
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
    model.to(device)
    
    # Setup optimizer and loss function
    optimizer = LBFGS(model.parameters(), lr=config.training.learning_rate)
    # optimizer = Adam(model.parameters(), lr=config.training.learning_rate)
    loss_fn = MSELoss(reduction="mean")

    # Load data
    data = pd.read_csv(DATA_PATH / "train.csv", parse_dates=["start_time"])
    validation_data = pd.read_csv(DATA_PATH / "validation.csv", parse_dates=["start_time"])

    # Fit preprocesser to train data, and transform test data
    preprocesser = Preprocesser(
        time_of_day=config.data.time_of_day,
        time_of_week=config.data.time_of_week,
        time_of_year=config.data.time_of_year,
        last_day_y=config.data.last_day_y,
        two_last_day_y=config.data.two_last_day_y
    )
    processed = preprocesser.fit_transform(data)
    validation_processed = preprocesser.transform(validation_data)

    # Split train data, convert to tensors
    inputs_df, target_df = split_data(processed)
    inputs = pd_to_tensor(inputs_df)
    targets = pd_to_tensor(target_df)

    # Split validation data, convert to tensors
    validation_inputs_df, validation_target_df = split_data(validation_processed)
    validation_inputs = pd_to_tensor(validation_inputs_df)
    validation_targets = pd_to_tensor(validation_target_df)

    # Create train dataset and dataloader
    dataset = TimeSeriesDataset(inputs, targets, window=config.data.sequence_length)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True,
        # Move both training and test data to device, for CUDA training
        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
    )

    # Create validation dataset and dataloader
    validation_dataset = TimeSeriesDataset(validation_inputs, validation_targets, window=config.data.sequence_length)
    validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False,
        # Move both training and test data to device, for CUDA training
        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
    )

    # Create combined giga-plot, for use later
    fig, axes = plt.subplots(
        nrows=config.training.epochs + 1,
        ncols=1,
        figsize=(20, (config.training.epochs + 1) * 10)
    )
    train_losses = [_get_validation_loss(model, dataloader, loss_fn)]
    validation_losses = [_get_validation_loss(model, validation_dataloader, loss_fn)]
    ax = axes.flat

    # Train model and plot with frequency save_frequency
    for epoch in tqdm(range(1, config.training.epochs + 1), unit="epoch"):
        _train_epoch(model, dataloader, optimizer, loss_fn)

        if epoch % config.training.save_frequency == 0:
            loss = _get_validation_loss(model, validation_dataloader, loss_fn)
            tqdm.write(f"Loss epoch {epoch}: \t {loss}")
            train_losses.append(_get_validation_loss(model, dataloader, loss_fn))
            validation_losses.append(loss)

            plot_validation_prediction(
                model, validation_dataloader, epoch,
                postprocess_target=preprocesser.reverse_y,
                base_save_path=base_save_path
            )

            # Add plot to combined plot
            plot_validation_prediction(
                model, validation_dataloader, epoch,
                postprocess_target=preprocesser.reverse_y,
                ax=ax[epoch]
            )

            model_savepath = base_save_path  / f"models/LSTM_{epoch}.pt"
            model.save_model(model_savepath)

    # Plot loss data to gigaplot
    plot_loss_data(validation_losses, ax[0], log_y=True, label="Validation loss")
    plot_loss_data(train_losses, ax[0], log_y=True, label="Training loss")

    # Plot loss data as own plot, as well
    loss_fig, loss_ax = plt.subplots(1, 1, figsize=(20, 10))
    plot_loss_data(validation_losses, loss_ax, log_y=True, label="Validation loss")
    plot_loss_data(train_losses, loss_ax, log_y=True, label="Training loss")
    _save_losses(validation_losses, base_save_path / "validation_losses")
    _save_losses(train_losses, base_save_path / "train_losses")

    plt.tight_layout()
    fig.savefig(base_save_path / "LSTM.png")
    loss_fig.savefig(base_save_path / "loss.png")
    plt.close(fig)
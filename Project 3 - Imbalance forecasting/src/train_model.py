import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import MSELoss
from torch.optim import Adam, LBFGS
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from src.helpers.config import read_config
from src.helpers.dataset import TimeSeriesDataset
from src.helpers.path import DATA_PATH, ROOT_PATH
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
    for val_input, val_target in dataloader:
        val_output = model(val_input)
        loss = loss_fn(val_output, val_target)

        losses.append(loss)
    
    return sum(losses) / len(losses)


def _plot_validation_prediction(model,
                                dataloader,
                                epoch: int,
                                postprocess_target = lambda x: x,
                                in_ax: plt.Axes = None
    ):
    if ax is None:
        create_new = True
        savepath = ROOT_PATH / f"plots/training/LSTM_{epoch}.png"
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    else:
        create_new = False

    ax.set_title(f"Epoch {epoch}, single day prediction")

    for inputs, targets in dataloader:
        outputs = model(inputs)

        ax.plot(
            postprocess_target(outputs.detach().numpy(), label="Model outputs")
        )
        ax.plot(
            postprocess_target(targets.detach().numpy(), label="Targets")
        )

    ax.set_xticks([])
    ax.legend()
    
    if create_new:
        fig.savefig(savepath)
        plt.close(fig)



def train_model():
    # Load configs
    config = read_config(ROOT_PATH / "config.yml")

    # Setup model
    model = LSTM(
        input_size = 8,
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
    preprocesser = Preprocesser()
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

    # Train model and plot with frequency save_frequency
    for epoch in tqdm(range(1, config.training.epochs + 1), unit="epoch"):
        _train_epoch(model, dataloader, optimizer, loss_fn)

        if epoch % config.training.save_frequency == 0:
            loss = _get_validation_loss(model, validation_dataloader, loss_fn)
            tqdm.write(f"Loss epoch {epoch}: \t {loss}")

            _plot_validation_prediction(model, validation_dataloader, epoch)

            model_savepath = ROOT_PATH / f"models/training/LSTM_{epoch}.pt"
            model.save_model(model_savepath)
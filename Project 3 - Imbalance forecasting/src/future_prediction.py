import torch
from torch.utils.data import DataLoader
from helpers.dataset import FutureDataset

from src.lstm import LSTM

def predict_into_future(model: LSTM,
                        dataloader: DataLoader,
                        timesteps_into_future: int
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
                    (timestep_prediction, timestep_target)
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



if __name__ == "__main__":
    model = LSTM(3, 2, 2)
    inputs = torch.tensor([list(range(9)), list(range(10, 19)), list(range(20, 29))], dtype=float).T
    targets = torch.tensor(list(range(90, 99)), dtype=float).reshape((-1, 1))
    dataset = FutureDataset(inputs, targets, past_window=2, future_window=2)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    print(predict_into_future(model, dataloader, 2))
import torch
from torch import nn

class LSTM(nn.Module):
    """LSTM class

    Consists of a user-selected number of LSTMCells, each with a user-selected
    number of hidden layers (both selected in __init__).

    The data used in .forward must be of the following size:
        (N, L, input_size),
    where
        N = the number of sequences in each batch of data, and
        L = the number of timesteps in each sequence
    
    """
    def __init__(self, input_size: int, lstm_depth: int, hidden_layers: int) -> None:
        """Create an instance of an LSTM class

        Args:
            input_size (int): Number of features the LSTM takes as input for
                each step in the sequence
            lstm_depth (int): The number of LSTMCells should be used
            hidden_layers (int): The number of hidden layers in each LSTMCell
        """
        super().__init__()

        self.hidden_layers = hidden_layers
        
        cells = [nn.LSTMCell(input_size, hidden_layers, dtype=float)]
        for i in range(lstm_depth - 1):
            cell = nn.LSTMCell(hidden_layers, hidden_layers, dtype=float)
            cells.append(cell)

        self.cells = nn.ModuleList(cells)
        self.output_layer = nn.Linear(hidden_layers, 1, dtype=float)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict output, using x as input.

        Args:
            x (torch.Tensor): Tensor of input-data. Must have shape
                    (N, L, input_size),
                where
                    N = the number of sequences in each batch of data, and
                    L = the number of timesteps in each sequence

        Returns:
            Tensor: Predicted output for each timestep. Has shape (N, 1)
        """
        # x has shape (number of samples, sample length, input_size)
        batch_size = x.size(0)

        hidden_states = [torch.zeros(batch_size, self.hidden_layers, dtype=float) for _ in range(len(self.cells))]
        cell_states = [torch.zeros(batch_size, self.hidden_layers, dtype=float) for _ in range(len(self.cells))]

        for time_step in x.split(1, dim=1):
            # time_step is each step in each sequence/sample.
            # If N sequences are given, each with length = 10, then we iterate 10 times, with
            #   time_step.shape == (N, 1, input_size)
            # Convert this to
            #   time_step.shape == (N, input_size)
            # which is what LSTMCell wants to receive
            time_step = torch.squeeze(time_step, dim=1)

            # Compute value from first layer separately
            hidden_states[0], cell_states[0] = self.cells[0](time_step)
            cell_input = hidden_states[0]

            for i, cell in enumerate(self.cells[1:], start=1):
                # Use previous layer's hidden and previous time_step's cell
                # state to compute value from each subsequent layers
                hidden_state, cell_state = cell(cell_input, (hidden_states[i - 1], cell_states[i]))

                # Update states and cell_input
                hidden_states[i] = hidden_state
                cell_states[i] = cell_state
                cell_input = hidden_state
        
        # The last hidden_state is now our output from the LSTMCells
        return self.output_layer(hidden_state)
    
    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()



if __name__ == "__main__":
    lstm = LSTM(input_size=5, lstm_depth=3, hidden_layers=10)
    print(lstm)
    
    output_data = torch.rand((10))

    data = torch.rand((2, 10, 5))
    print(lstm(data))

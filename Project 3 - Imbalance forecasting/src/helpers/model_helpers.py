from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


def train_model(model: Module,
                data_loader: DataLoader,
                optimizer,
                epochs: int,
                loss_function,
                epoch_print_frequency: int = 1):
    for epoch in trange(1, epochs + 1, unit="epoch"):
        losses = []

        for inputs, targets in tqdm(data_loader, unit="batch", leave=False):
            predicted = model(inputs)
            loss = loss_function(predicted, targets)
            losses.append(loss)

            # Reset gradients of model's parameters
            optimizer.zero_grad()
            # Backpropagate loss
            loss.backward()
            # Adjust parameters by the gradients collected in backward pass
            optimizer.step()
        
        if epochs % epoch_print_frequency == 0:
            tqdm.write(f"Average epoch loss (epoch {epoch - epoch_print_frequency}-{epoch}): {sum(losses) / len(losses)}")
            losses = []


from src.autoencoder import AutoEncoder, device
from supplied_files.stacked_mnist import DataMode, StackedMNISTData
from supplied_files.verification_net import VerificationNet

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

def train_autoencoder(
        datamode: DataMode,
        channels: int,
        model_save_path: str,
        predictability_tolerance: float = 0.8,
        learning_rate: float = 1e-3,
        epochs: int = 500):

    model = AutoEncoder(channels=channels).to(device)
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    data_generator = StackedMNISTData(datamode, default_batch_size=10000)
    verification_net = VerificationNet()

    for epoch in range(epochs):
        print(f"\n########## EPOCH {epoch + 1} ##########")

        ### TRAINING ###
        model.train()
        for X, y in data_generator.batch_generator():
            X = X.to(device)
            
            X_hat = model(X)
            loss = loss_fn(X_hat, X)

            # Backpropagation - this time implemented by Pytorch (i.e. it works)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        ### TESTING ###
        model.eval()
        X_test, y_test = data_generator.get_full_data_set(training=False)
        X_test = X_test.to(device)
        X_hat_test = model(X_test)
        
        test_loss = loss_fn(X_hat_test, X_test)

        predictability_data = torch.permute(
            X_hat_test, (0, 2, 3, 1)
        ).cpu().detach().numpy()

        predictability, accuracy = verification_net.check_predictability(
            predictability_data, y_test, tolerance=0.5
        )
        print(f"Test loss:      {test_loss:.5f}")
        print(f"Predictability: {100*predictability:.2f}%")
        print(f"Accuracy:       {100*accuracy:.2f}%")
    
    torch.save(model.state_dict(), model_save_path)




if __name__ == "__main__":
    # Train single-layer image
    train_autoencoder(
        DataMode.MONO_BINARY_COMPLETE, 1, "models/autoencoder/mono.pt"
    )

    # Train multi-layer image
    train_autoencoder(
        DataMode.COLOR_BINARY_COMPLETE, 3, "models/autoencoder/color.pt", 0.5
    )

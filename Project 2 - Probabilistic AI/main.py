from src.autoencoder import AutoEncoder, device
from supplied_files.stacked_mnist import DataMode, StackedMNISTData
from supplied_files.verification_net import VerificationNet

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam


def train_autoencoder_mono():
    LEARNING_RATE = 1e-3
    DATAMODE = DataMode.MONO_BINARY_COMPLETE
    EPOCHS = 500

    model = AutoEncoder().to(device)
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    data_generator = StackedMNISTData(DATAMODE)
    verification_net = VerificationNet()

    for epoch in range(EPOCHS):
        print(f"\n########## EPOCH {epoch + 1} ##########")

        ### TRAINING ###
        model.train()
        X, y = data_generator.get_full_data_set()
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
        X_hat_test = model(X_test)
        
        test_loss = loss_fn(X_hat_test, X_test)

        predictability_data = torch.permute(
            X_hat_test, (0, 2, 3, 1)
        ).detach().numpy()

        predictability, accuracy = verification_net.check_predictability(
            predictability_data, y_test, tolerance=0.8
        )
        print(f"Test loss:      {test_loss:.5f}")
        print(f"Predictability: {100*predictability:.2f}%")
        print(f"Accuracy:       {100*accuracy:.2f}%")
    
    torch.save(model.state_dict(), "models/autoencoder/mono.pt")


def train_autoencoder_color():
    LEARNING_RATE = 1e-3
    DATAMODE = DataMode.COLOR_BINARY_COMPLETE
    EPOCHS = 500

    model = AutoEncoder(channels=3).to(device)
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    data_generator = StackedMNISTData(DATAMODE, default_batch_size=10000)
    verification_net = VerificationNet()

    for epoch in range(EPOCHS):
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
    
    torch.save(model.state_dict(), "models/autoencoder/color.pt")




if __name__ == "__main__":
    train_autoencoder_color()

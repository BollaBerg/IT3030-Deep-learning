from src.autoencoder import AutoEncoder, device
from supplied_files.stacked_mnist import DataMode, StackedMNISTData
from supplied_files.verification_net import VerificationNet

import matplotlib.pyplot as plt
import torch
from torch.nn import BCELoss
from torch.optim import Adam

def load_model(model, model_path: str):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def train_autoencoder(
        datamode: DataMode,
        channels: int,
        model_save_path: str,
        predictability_tolerance: float = 0.8,
        learning_rate: float = 1e-3,
        epochs: int = 500,
        batch_size: int = 10000,
        plot_savepath: str = "images/autoencoder/training.png"):

    model = AutoEncoder(channels=channels).to(device)
    loss_fn = BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    data_generator = StackedMNISTData(datamode, default_batch_size=batch_size)
    verification_net = VerificationNet(file_name="./models/verification/verification_model")
    verification_net.load_weights()
    
    accuracies = []
    predictabilities = []

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
        y_test = y_test.cpu().detach().numpy()

        predictability, accuracy = verification_net.check_predictability(
            predictability_data, y_test, tolerance=predictability_tolerance
        )
        print(f"Test loss:      {test_loss:.5f}")
        print(f"Predictability: {100*predictability:.2f}%")
        print(f"Accuracy:       {100*accuracy:.2f}%")

        predictabilities.append(predictability)
        accuracies.append(accuracy)

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                f"models/autoencoder/training/epoch_{epoch}_{test_loss}.pt"
            )
            plot_model_results(
                X_test, X_hat_test, 10, channels, f"images/autoencoder/training/epoch_{epoch}.png"
            )
    
    torch.save(model.state_dict(), model_save_path)

    plot_predictabilies_accuracies(predictabilities, accuracies, plot_savepath)


def plot_predictabilies_accuracies(predictabilities, accuracies, plot_savepath):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(predictabilities, label="Predictability")
    ax.plot(accuracies, label="Accuracy")

    fig.legend()
    fig.suptitle("Predictability and accuracy - AE training", fontsize=48)
    plt.tight_layout()
    plt.savefig(plot_savepath)


def plot_model_results(mnist_batch,
                       generated_batch,
                       num_images: int,
                       channels: int,
                       plot_savepath: str):
    generated_batch = torch.permute(generated_batch, (0, 2, 3, 1)).cpu().detach().numpy()
    mnist_batch = torch.permute(mnist_batch, (0, 2, 3, 1)).cpu().detach().numpy()

    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(20, 10*num_images))
    for row in range(num_images):
        if channels == 1:
            axes[row][0].imshow(mnist_batch[row, :, :, 0], cmap="binary")
            axes[row][1].imshow(generated_batch[row, :, :, 0], cmap="binary")
        else:
            axes[row][0].imshow(mnist_batch[row, :, :, :].astype(float))
            axes[row][1].imshow(generated_batch[row, :, :, :].astype(float))
        
        axes[row][0].set_xticks([])
        axes[row][0].set_yticks([])
        axes[row][1].set_xticks([])
        axes[row][1].set_yticks([])

        # axes[row][0].set_title(str(labels[row].item()))
        # axes[row][1].set_title(str(labels[row].item()))
    
    fig.suptitle("Examples of model performance", fontsize=48)
    fig.tight_layout()
    # plt.show()
    plt.savefig(plot_savepath)
    print(f"Plot saved at {plot_savepath}")


def plot_autoencoder_result(
        datamode: DataMode,
        channels: int,
        model_path: str,
        num_images: int = 10,
        plot_savepath: str = "images/autoencoder_results.png"
        ):
    data_generator = StackedMNISTData(datamode)
    model = AutoEncoder(channels=channels)
    model = load_model(model, model_path)
    model.eval()

    mnist_batch, labels = data_generator.get_random_batch(training=False, batch_size=num_images)
    generated_batch = model(mnist_batch)

    plot_model_results(mnist_batch, generated_batch, num_images, channels, plot_savepath)


def plot_color_result_individually(
        model_path: str,
        plot_savepath: str,
        num_images: int = 10,
        ):
    data_generator = StackedMNISTData(DataMode.COLOR_BINARY_COMPLETE)
    model = AutoEncoder(channels=3)
    model = load_model(model, model_path)
    model.eval()

    mnist_batch, labels = data_generator.get_random_batch(training=False, batch_size=num_images)
    generated_batch = model(mnist_batch)

    generated_batch = torch.permute(generated_batch, (0, 2, 3, 1)).detach().numpy()
    mnist_batch = torch.permute(mnist_batch, (0, 2, 3, 1)).detach().numpy()

    fig, axes = plt.subplots(nrows=num_images, ncols=4, figsize=(20, 5*num_images))
    for row in range(num_images):
        axes[row][0].imshow(mnist_batch[row, :, :, :].astype(float))
        axes[row][1].imshow(generated_batch[row, :, :, 0], cmap="binary")
        axes[row][2].imshow(generated_batch[row, :, :, 1], cmap="binary")
        axes[row][3].imshow(generated_batch[row, :, :, 2], cmap="binary")
        
        for i in range(4):
            axes[row][i].set_xticks([])
            axes[row][i].set_yticks([])
    
    fig.suptitle("AE on color data - each color plotted individually", fontsize=48)
    # plt.show()
    plt.savefig(plot_savepath)
    print(f"Plot saved at {plot_savepath}")


def print_model_output(model_path: str, channels: int):
    if channels == 1:
        datamode = DataMode.MONO_BINARY_COMPLETE
    else:
        datamode = DataMode.COLOR_BINARY_COMPLETE

    data_generator = StackedMNISTData(datamode)
    model = AutoEncoder(channels=channels)
    model = load_model(model, model_path)
    model.eval()

    data, _ = data_generator.get_random_batch(training=False, batch_size=1)
    output = model(data)
    print(output)


def plot_generative_model(model_path: str, channels: int, plot_savepath: str):
    if channels == 1:
        datamode = DataMode.MONO_BINARY_COMPLETE
    else:
        datamode = DataMode.COLOR_BINARY_COMPLETE

    data_generator = StackedMNISTData(datamode)
    model = AutoEncoder(channels=channels)
    model = load_model(model, model_path)
    model.eval()

    verification_net = VerificationNet(file_name="./models/verification/verification_model")
    verification_net.load_weights()

    num_generatives = 10000
    pred_tolerance = 0.8 if channels == 1 else 0.5
    class_tolerance = 0.95 if channels == 1 else 0.9

    data, _ = data_generator.get_full_data_set(training=False)
    encoded = model.encode(data.reshape(-1, 1, 28, 28))
    maxes = torch.max(encoded, 0).values
    mins = torch.min(encoded, 0).values

    z = (maxes - mins) * torch.rand((channels * num_generatives, 28)) + mins
    outputs = model.decode(z)
    outputs = outputs.reshape(-1, channels, 28, 28)
    images = torch.permute(outputs, (0, 2, 3, 1)).detach().numpy()

    predictability, _ = verification_net.check_predictability(
        images, tolerance=pred_tolerance
    )
    coverage = verification_net.check_class_coverage(data=images, tolerance=class_tolerance)

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
    axes = axes.flat
    for i in range(16):
        if channels == 1:
            axes[i].imshow(images[i, :, :, 0], cmap="binary")
        else:
            axes[i].imshow(images[i, :, :, :].astype(float))

        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    fig.suptitle("AE as generative model", fontsize=48)
    axes[12].text(0.5, 0.0, 
        f"{num_generatives} generated images",
        horizontalalignment="center", verticalalignment="top",
        transform=axes[12].transAxes,
        fontsize=20
    )
    axes[13].text(0.5, 0.0, 
        f"Predictability: {100*predictability:.2f}%",
        horizontalalignment="center", verticalalignment="top",
        transform=axes[13].transAxes,
        fontsize=20
    )
    axes[14].text(0.5, 0.0,
        f"Coverage: {100*coverage:.2f}%",
        horizontalalignment="center", verticalalignment="top",
        transform=axes[14].transAxes,
        fontsize=20
    )
    axes[15].text(0.5, 0.0, 
        f"Pred.tol.: {pred_tolerance}, class tol.: {class_tolerance}",
        horizontalalignment="center", verticalalignment="top",
        transform=axes[15].transAxes,
        fontsize=20
    )
    fig.tight_layout()
    # plt.show()
    plt.savefig(plot_savepath)
    print(f"Generative mode saved at {plot_savepath}")


def plot_anomaly_detection(model_path: str, channels: int, plot_savepath: str):
    if channels == 1:
        datamode = DataMode.MONO_BINARY_COMPLETE
    else:
        datamode = DataMode.COLOR_BINARY_COMPLETE

    data_generator = StackedMNISTData(datamode)
    loss_fn = BCELoss()
    model = AutoEncoder(channels=channels)
    model = load_model(model, model_path)
    model.eval()

    data, _ = data_generator.get_full_data_set(training=False)
    data_hat = model(data)

    outputs = []
    for i in range(len(data)):
        outputs.append((data[i], loss_fn(data[i], data_hat[i])))
    
    outputs.sort(key=lambda x: x[1], reverse=True)
    
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
    axes = axes.flat
    for i in range(16):
        image = torch.permute(outputs[i][0], (1, 2, 0)).cpu().detach().numpy()
        if channels == 1:
            axes[i].imshow(image[:, :, 0], cmap="binary")
        else:
            axes[i].imshow(image[:, :, :].astype(float))

        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(f"Loss: {outputs[i][1]}")
    
    fig.suptitle("AE anomaly detector", fontsize=48)
    fig.tight_layout()
    plt.savefig(plot_savepath)
    print(f"Anomaly detection saved at {plot_savepath}")
    


if __name__ == "__main__":
    # # Train single-layer image
    # train_autoencoder(
    #     DataMode.MONO_BINARY_COMPLETE, 1, "models/autoencoder/mono.pt",
    #     epochs=100
    # )

    # # NOT NEEDED, USE SINGLE-LAYER MODEL TO PREDICT COLORS AS WELL :)
    # # Train multi-layer image
    # train_autoencoder(
    #     DataMode.COLOR_BINARY_COMPLETE, 3, "models/autoencoder/color.pt", 0.5,
    #     epochs=100,
    #     batch_size = 60000
    # )
    
    # # Plot single-layer results
    # plot_autoencoder_result(
    #     DataMode.MONO_BINARY_COMPLETE, 1, "models/autoencoder/mono_demo.pt",
    #     plot_savepath="images/autoencoder/mono.png", num_images=10
    # )

    # # Plot multi-layer results
    # plot_autoencoder_result(
    #     DataMode.COLOR_BINARY_COMPLETE, 3, "models/autoencoder/mono_demo.pt",
    #     plot_savepath="images/autoencoder/color.png", num_images=10
    # )

    # # Plot multi-layer results individually
    # plot_color_result_individually(
    #     "models/autoencoder/mono_demo.pt", "images/autoencoder/color_individually.png"
    # )

    # # Print model results
    # print_model_output("models/autoencoder/mono_demo.pt", 1)
    # print_model_output("models/autoencoder/mono_demo.pt", 3)

    # # Plot AE as generative model - single color
    # plot_generative_model(
    #     "models/autoencoder/mono_demo.pt", 1, "images/autoencoder/generative_mode.png"
    # )
    # Plot AE as generative model - multicolor
    plot_generative_model(
        "models/autoencoder/mono_demo.pt", 3, "images/autoencoder/generative_mode_color.png"
    )

    # # Train anomaly detector
    # train_autoencoder(
    #     DataMode.MONO_BINARY_MISSING, 1, "models/autoencoder/mono_anomaly.pt",
    #     epochs=100
    # )
    # # Use AE as anomaly detector - single color
    # plot_anomaly_detection(
    #     "models/autoencoder/mono_anomaly.pt", 1, "images/autoencoder/anomaly_detection.png"
    # )
    # # Use AE as anomaly detector - multicolor
    # plot_anomaly_detection(
    #     "models/autoencoder/mono_anomaly.pt", 3, "images/autoencoder/anomaly_detection_color.png"
    # )

import matplotlib.pyplot as plt
from torch.nn import MSELoss
from tqdm import tqdm

from src.future_prediction import predict_future_from_paths
from src.helpers.path import DATA_PATH, ROOT_PATH



def plot_prediction_losses(
            model_list: list,
        ):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    loss_fn = MSELoss(reduction="mean")
    cmap = plt.get_cmap("Set1").colors

    for i, model in enumerate(tqdm(model_list, unit="model")):
        name, config_path, model_path = model

        predictions = predict_future_from_paths(
            config_path, model_path,
            data_path=DATA_PATH / "validation.csv",
            future_steps=int(120 / 5),
        )

        losses = [
            loss_fn(pred[0], pred[1]).item() for pred in predictions
        ]
        ax.plot(losses, color=cmap[i], label=name)
    
    ax.set_title("Loss per future step")
    ax.set_xlabel("Steps into the future")
    ax.set_ylabel("log(loss) [MSE of actual, unscaled data]")
    ax.set_yscale("log")
    ax.legend()

    savepath = ROOT_PATH / "plots/model_future_differences.png"
    fig.savefig(savepath)
    print(f"Model prediction differences saved to {savepath}")



if __name__ == "__main__":
    paths = [
        ("All options off", "models/configs/1_all_off.yml", "models/1_all_off.pt"),
        ("Time columns on", "models/configs/2_time_on.yml", "models/2_time_on.pt"),
        ("Past targets on", "models/configs/3_hanging_y.yml", "models/3_hanging_y.pt"),
        ("All extras on", "models/configs/4_all_on.yml", "models/4_all_on.pt"),
        ("Time of day and week", "models/configs/5_day_week.yml", "models/5_day_week.pt"),
        ("Time of day, year + past targets", "models/configs/6_day_year_hanging.yml", "models/6_day_year_hanging.pt"),
        ("Time of day, year + past targets (30 steps)", "models/configs/7_day_year_targets_more_memory.yml", "models/7_day_year_targets_more_memory.pt"),
        ("Randomness in last_y", "models/configs/8_randomness_last_y.yml", "models/8_randomness_last_y.pt"),
    ]
    plot_prediction_losses(paths)

import pathlib
import sys

from src.demo import demo
from src.helpers.config import read_config
from src.helpers.path import DATA_PATH, ROOT_PATH
from src.train_model import train_model

def print_help():
    print("IMBALANCE FORECASTER")
    print()
    print("USAGE:")
    print("    python main.py [ARGS]")
    print()
    print("ARGUMENTS:")
    print("     --help | -h")
    print("         Print this help")
    print("     --train | -t [BASE_PATH]")
    print("         Train a model.")
    print("         Args:")
    print("             BASE_PATH (optional): Path where the config should be")
    print("                 loaded from, and where the model and plot will be")
    print("                 saved. Needs to contain the following:")
    print("                     config.yml")
    print("                     models/")
    print("                     plots/")
    print("                 Defaults to models/training")
    print("     --demo | -d CONFIG_PATH MODEL_PATH DATA_PATH")
    print("         Demo a model")
    print("         Args:")
    print("             DATA_PATH (optional): Path to data file. Defaults to")
    print("                 data/validation.csv")
    print("             CONFIG_PATH (optional): Path to config file. Defaults")
    print("                 to best trained model.")
    print("             MODEL_PATH (optional): Path of saved model. Defaults")
    print("                 to best trained model")


if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1] in ["--help", "-h"]:
        print_help()
        exit(0)

    if sys.argv[1] in ["--train", "-t"]:
        if len(sys.argv) > 2:
            base_path = pathlib.Path(sys.argv[2])
            config_path = base_path / "config.yml"
        else:
            base_path = pathlib.Path("models/training")
            config_path = ROOT_PATH / "config.yml"
        
        config = read_config(config_path)
        print(f"Loaded config from {config_path}")
        train_model(config, base_path)
    
    elif sys.argv[1] in ["--demo", "-d"]:
        if len(sys.argv) >= 3:
            data_path = pathlib.Path(sys.argv[2])
        else:
            data_path = DATA_PATH / "validation.csv"
        
        if len(sys.argv) >= 4:
            config_path = pathlib.Path(sys.argv[3])
        else:
            config_path = ROOT_PATH / "models/configs/7_day_year_targets_more_memory.yml"

        if len(sys.argv) >= 5:
            model_path = pathlib.Path(sys.argv[4])
        else:
            model_path = ROOT_PATH / "models/7_day_year_targets_more_memory.pt"
        
        demo(config_path, model_path, data_path)

    else:
        print_help()

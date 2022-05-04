import pathlib
import sys

from src.helpers.config import read_config
from src.helpers.path import ROOT_PATH
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
    print("     --train | -t")
    print("         Train a model. Save the trained model to")
    print("         models/training/LSTM_{epoch}.pt")


if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1] in ["--help", "-h"]:
        print_help()
        exit(0)

    if sys.argv[1] in ["--train", "-t"]:
        if len(sys.argv) > 3:
            base_path = pathlib.Path(sys.argv[3])
            config_path = base_path / "config.yml"
        else:
            base_path = pathlib.Path("models/training")
            config_path = ROOT_PATH / "config.yml"
        
        config = read_config(config_path)
        train_model(config, base_path)
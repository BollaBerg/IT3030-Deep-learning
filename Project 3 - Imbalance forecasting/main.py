import sys

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
    if len(sys.argv) <= 1 or "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        exit(0)
    if "--train" in sys.argv or "-t" in sys.argv:
        train_model()
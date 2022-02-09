import argparse

import matplotlib.pyplot as plt

from src.config import read_config
from src.generator import demo_generator

def parse() -> str:
    parser = argparse.ArgumentParser(
        description="Andreas B. Berg's attempt at backpropagation"
    )

    parser.add_argument(
        "path", type=str,
        help="Path to config file that should be read"
    )
    parser.add_argument(
        "-g", "--generator", dest="demo_generator", action="store_true",
        help="Demonstrate generator and quit the program (no Network running)"
    )
    parser.set_defaults(demo_generator=False)

    args = parser.parse_args()
    return args


def main():
    args = parse()

    if args.demo_generator:
        demo_generator()
        return

    network, datasets, epochs = read_config(args.path)
    train_losses, validation_losses = network.train(datasets[0], epochs=epochs, validation_set=datasets[1])
    test_losses = network.test(datasets[2])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.plot(train_losses.flatten(), fmt="-b")
    plt.plot(validation_losses.flatten(), fmt="-y")
    plt.plot(test_losses.flatten(), fmt="--m")

    plt.show()
    plt.savefig("images/loss_progression.png")

if __name__ == "__main__":
    main()
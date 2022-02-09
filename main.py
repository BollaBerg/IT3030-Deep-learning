import argparse
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

    network, datasets = read_config(args.path)


if __name__ == "__main__":
    main()
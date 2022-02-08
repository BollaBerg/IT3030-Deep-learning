import pathlib
import yaml

from network import Network

DEFAULTS = {
    "lrate" : 0.1,
    "wrt" : None,
    "wreg" : 0.001,
    "bias_range" : [0, 1],
    "distribution" : [70, 20, 10],
    "noise" : 0.0,
    "centering" : 0.0,
    "softmax": False,
}

def read_config(path : str | pathlib.Path) -> Network:
    file = pathlib.Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Couldn't find file at {path}")
    

    with file.open("r") as f:
        config = yaml.safe_load(f)
    
    layers = config.get("layers")
    
    # Handle input layer
    input_size = layers.pop(0).get("input", None)
    if input_size is None:
        raise ValueError("First layer must follow syntax: `layer: size`")
    
    # Handle (optional) softmax layer
    softmax = False
    if layers[-1].get("type", None) is not None:
        final_layer = layers.pop(-1).get("type")
        if final_layer != "softmax":
            raise ValueError("Type can only be `softmax` or not used")
        else:
            softmax = True

    for layer in layers:
        print(layer)


if __name__ == "__main__":
    read_config("configs/example_config.yaml")
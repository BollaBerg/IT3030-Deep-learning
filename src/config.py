from collections.abc import Iterable
from email import generator
from multiprocessing.sharedctypes import Value
import pathlib

import yaml

from src.activation_functions import Sigmoid, Tanh, Relu, Linear
from src.regularization import L1, L2, NoRegularization
from src.generator import Image, Generator
from src.layer import Layer
from src.loss_functions import CrossEntropy, MeanSquaredError
from src.network import Network

DEFAULTS = {
    "lrate" : 0.1,
    "wrt" : None,
    "wreg" : 0.001,
    "bias_range" : [0, 1],
    "distribution" : [70, 20, 10],
    "noise" : 0.0,
    "centering" : 0.0,
    "softmax": False,
    "debug": False,
    "verbose": False,
    "epochs": 10
}

def read_config(path : str | pathlib.Path) -> tuple[Network, list[list[Image]], int]:
    file = pathlib.Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Couldn't find file at {path}")
    

    with file.open("r") as f:
        config = yaml.safe_load(f)
    
    ### DEFAULTS ###
    loss_name = config.get("loss", "")
    if loss_name.lower() == "cross_entropy":
        loss_function = CrossEntropy()
    elif loss_name.lower() in ("mse", "meansquarederror"):
        loss_function = MeanSquaredError()
    else:
        raise ValueError(
            "Loss must be one of 'cross_entropy' or 'mse'. "
            f"Loss was {loss_name}"
        )
    
    learning_rate = config.get("lrate", DEFAULTS.get("lrate"))
    DEFAULTS["lrate"] = learning_rate

    weight_regularization_rate = config.get("wreg", DEFAULTS.get("wreg"))

    weight_reg_name = config.get("wrt", DEFAULTS.get("wrt"))
    if weight_reg_name is None or weight_reg_name.lower() == "none":
        weight_regularization = NoRegularization(weight_regularization_rate)
    elif weight_reg_name.lower() == "l1":
        weight_regularization = L1(weight_regularization_rate)
    elif weight_reg_name.lower() == "l2":
        weight_regularization = L2(weight_regularization_rate)
    else:
        raise ValueError(
            f"wrt must be one of 'l1', 'l2' or 'none'. wrt was {weight_reg_name}"
        )

    debug = config.get("debug", DEFAULTS.get("debug"))
    verbose = config.get("verbose", DEFAULTS.get("verbose"))
    epochs = config.get("epochs", DEFAULTS.get("epochs"))

    ### LAYERS ###

    layers = config.get("layers")
    
    # Handle input layer
    input_size = layers.pop(0).get("input", None)
    if input_size is None:
        raise ValueError("First layer must follow syntax: `layer: size`")
    
    # Handle (optional) softmax layer
    if layers[-1].get("softmax", False):
        softmax = True
        layers.pop(-1)
    elif layers[-1].get("softmax", None) is not None:
        softmax = False
        layers.pop(-1)
    else:
        softmax = False

    network_layers = []
    for i, layer in enumerate(layers):
        size = layer.get("size", None)
        if size is None:
            raise ValueError(
                f"All layers must have a size! Layer {i} does not have size."
            )
        
        activation_function_name = layer.get("act", "")
        if activation_function_name.lower() == "sigmoid":
            activation_function = Sigmoid()
        elif activation_function_name.lower() == "tanh":
            activation_function = Tanh()
        elif activation_function_name.lower() == "relu":
            activation_function = Relu()
        elif activation_function_name.lower() == "linear":
            activation_function = Linear()
        else:
            raise ValueError(
                "Activation function must be legal value. "
                f"Function for layer {i} was {activation_function_name}"
            )

        weight_range_input = layer.get("wr", None)
        if isinstance(weight_range_input, str) and weight_range_input.lower() == "glorot":
            weight_range = "glorot"
        elif isinstance(weight_range_input, list) and len(weight_range_input) == 2:
            weight_range = (weight_range_input[0], weight_range_input[1])
        else:
            raise ValueError(
                f"Weight range (wr) was illegal value for layer {i}: {weight_range_input}"
            )
        
        learning_rate = layer.get("lrate", DEFAULTS.get("lrate"))

        bias_range = layer.get("br", DEFAULTS.get("bias_range"))
        if not isinstance(bias_range, Iterable) or not len(bias_range) == 2:
            raise ValueError(
                "Bias range (br) must be an iterable with length 2. "
                f"Got, for node {i}: {bias_range}"    
            )
        
        if len(network_layers) < 1:
            prev_size = input_size
        else:
            prev_size = network_layers[-1].biases.size

        network_layers.append(Layer(
            size=size,
            input_size=prev_size,
            activation_function=activation_function,
            initial_weight_range=weight_range,
            learning_rate=learning_rate,
            bias_range=bias_range,
        ))

    network = Network(
        input_size=input_size,
        loss_function=loss_function,
        weight_regularization=weight_regularization,
        layers=network_layers,
        softmax=softmax,
        debug=debug,
        verbose=verbose,
    )

    ### DATASET ###
    dataset = config.get("dataset")
    generator = Generator()
    generated_datasets = []
    if dataset.get("training_path", None) is not None:
        training_path = dataset.get("training_path")
        training_set = generator.read_file(training_path)
        generated_datasets.append(training_set)
    if dataset.get("validation_path", None) is not None:
        validation_path = dataset.get("validation_path")
        validation_set = generator.read_file(validation_path)
        generated_datasets.append(validation_set)
    if dataset.get("test_path", None) is not None:
        test_path = dataset.get("test_path")
        test_set = generator.read_file(test_path)
        generated_datasets.append(test_set)
    
    if len(generated_datasets) > 0:
        return network, generated_datasets, epochs
    
    dimension = dataset.get("dimension", None)
    if dimension is None:
        raise ValueError("dataset.validation must be a valid integer")
    
    number = dataset.get("number", None)
    if number is None:
        raise ValueError("dataset.number must be a valid integer")
    
    flatten = dataset.get("flatten", False)
    distribution = dataset.get("distribution", DEFAULTS.get("distribution"))
    noise = dataset.get("noise", DEFAULTS.get("noise"))
    centering = dataset.get("centering", DEFAULTS.get("centering"))

    generated_datasets = generator.get_multiple_sets(
        image_dimension=dimension,
        total_number_of_images=number,
        noise_portion=noise,
        set_distribution=distribution,
        centering_factor=centering,
        flatten=flatten
    )

    return network, generated_datasets, epochs




if __name__ == "__main__":
    read_config("configs/example_config.yaml")
# IT3030-Deep-learning
Coursework for the class IT3030 - Deep learning, taken during Spring 2022

## Environment
```python 3.10.2```

All requirements can be found in `requirements.txt`

## Structure
The structure is listed using the following system:
```
Optional Directory/
- file.py
    - Class
        "Description or other documentation"
        - methods
```

Note that this structure is mainly for helping myself plan the project, and while I will try to keep it up to date with the project, some changes may happen without me remembering to update this structure.
```
src/
 activation_functions.py
    "Activation functions that can be used in each Layer. Each function has two methods: `apply(x)` and `derivative(x)`
    - Sigmoid
    - Tanh
    - Relu
    - Linear
 config_options.py
    "Collection of enums of legal config options"
    - Loss(Enum)
    - WeightRegularization(Enum)
 config.py
    - ConfigReader
        "Reads user-supplied config files, returns a Network"
 generator.py
    -  Generator
        "Generates two-dimensional, binary pixel images"
        - get_single_set
        - get_multiple_sets
    - ImageClass (Enum)
        "Enum of possible image classes that can be generated"
    - Image (Dataclass)
        "Dataclass holding image data and image class"
 layer.py
    - Layer
        "Network layer, handling the core work of the Network"
        - forward_pass()
        - backward_pass()
 network.py
    - Network
        "Actual Neural/Deep Network, core of the project"
        - forward_pass()
        - backward_pass()
main.py
    "Main file, allowing the user to run Networks from a CLI, using configuration files"
```

## Config
Configuration is done through .yaml-files. Some examples of these can be found in the `configs/`-directory.

To explain possible configuration options, see the following example
```yaml
# configs/example_config.yaml
loss: cross_entropy
lrate: 0.1
wrt: L2
wreg: 0.001

layers:
  - input: 20
  - size: 100
    act: relu
    wr: [-0.1, 0.1]
    lrate: 0.01
  - size: 5
    act: relu
    wr: glorot
    br: [0, 1]
  - type: softmax

dataset:
    path: datasets/example.dataset
```

Anything defined initially (before `layers`) will be applied globally, and/or used as defaults if not overwritten in specific layers. The following options can be defined globally:

| Option | Description | Legal values | Mandatory |
| ------ | ----------- | ------------ | :-------: |
| `loss` | Function that will be used for calculating loss | `cross_entropy`, `MSE`| ✔️ |
| `lrate`| Default learning rate for the Network | Any value in the open interval `(0, 1)`| ❌ (defaults to `0.1`)|
| `wrt` | Global weight regularization | `L1`, `L2`, `none`| ❌ (defaults to `none`)|
| `wreg` | Global weight regularization rate | Any value in the open interval `(0, 1)` (usually small fractions) | ❌ (defaults to `0.001`) |
| `debug` | Whether the Network should run in DEBUG-mode, and log everything | Boolean | ❌ (defaults to False) |


Layers are defined in the `layers`-array. Each element is one layer. The following options can be defined for each layer:

| Option | Description | Legal values | Mandatory |
| ------ | ----------- | ------------ | :-------: |
| `input`| Number of input neurons. MUST be defined as the first layer, and only the first layer. Note that the input layer should not have any other options! | Any integer | ✔️ (as first layer only) |
| `size` | Size of the layer | Any integer in the interval `[1, 1000]` | ✔️ |
| `act`  | Activation function for the layer | `sigmoid`, `tanh`, `relu`, `linear` | ✔️ |
| `wr`   | Initial weight ranges for the layer | Either an inline array of `[min, max]`, or `glorot` to use the glorot initializer | ✔️ |
| `lrate`| Learning rate for the layer, if it is not using the default learning rate | Any value in the open interval `(0, 1)`| ❌ (defaults to the global `lrate`)|
| `br`   | Range for biases in the layer | Any value in the interval `[0, 1]` | ❌ (Defaults to `[0, 1]`) |
| `softmax` | Whether there should be a softmax layer after the last layer. ONLY allowed as last layer | Boolean | ❌ (defaults to `false`) |

Information about the dataset are defined in the `dataset` value. There are two ways to supply a dataset - through a path, or by giving options that will be used for generating the dataset when running the file. The example shows a path, but the following options can be used instead, to create the dataset when it is used:

| Option | Description | Legal values | Mandatory |
| ------ | ----------- | ------------ | :-------: |
| `dimension` | The height/width of the images that will be generated. Note that all images are square | Any integer in the interval `[10, 50]` | ✔️ |
| `number` | Total number of images in all datasets (i.e. the sum of training, validation and testing data) | Any integer | ✔️ |
| `flatten` | Whether the images should be flattened | Boolean | ✔️ |
| `distribution` | The distribution of images in the different sets | An array of integers (i.e. `[70, 20, 10]`) | ❌ (Defaults to `[70, 20, 10]`) |
| `noise` | The proportion of each image that should be noise | Any value in the interval `[0, 1]` | ❌ (Defaults to `0`) |
| `centering` | To what degree the figures should be centered in the image | Any value in the interval `[0, 1]` | ❌ (Defaults to `0`) |

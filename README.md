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
network.py
    - Network
        "Actual Neural/Deep Network, core of the project"
        - forward_pass()
        - backward_pass()
layer.py
    - Layer
        "Network layer, handling the core work of the Network"
        - forward_pass()
        - backward_pass()
generator.py
    -  Generator
        "Generates two-dimensional, binary pixel images"
        - get_training_set()
        - get_validation_set()
        - get_test_set()
        - get_all_sets()
    - ImageClass (Enum)
        "Enum of possible image classes that can be generated"
config.py
    - ConfigReader
        "Reads user-supplied config files, returns a Network"
main.py
    "Main file, allowing the user to run Networks from a CLI, using       configuration files"
```
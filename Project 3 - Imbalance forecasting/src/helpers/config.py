from dataclasses import dataclass
import pathlib

import yaml

@dataclass
class TrainingConfig:
    epochs: int
    save_frequency: int
    learning_rate: float

@dataclass
class DataConfig:
    sequence_length: int

@dataclass
class ModelConfig:
    lstm_depth: int
    hidden_layers: int

@dataclass
class Config:
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig


def read_config(path: str) -> tuple[TrainingConfig, DataConfig, ModelConfig]:
    file = pathlib.Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Couldn't find config at {path}")
    
    with file.open("r") as f:
        data = yaml.safe_load(f)
    
    training_config = TrainingConfig(
        epochs = data.get("training").get("epochs"),
        save_frequency = data.get("training").get("save_frequency"),
        learning_rate = data.get("training").get("learning_rate")
    )
    data_config = DataConfig(
        sequence_length = data.get("data").get("sequence_length"),
    )
    model_config = ModelConfig(
        lstm_depth = data.get("model").get("lstm_depth"),
        hidden_layers = data.get("model").get("hidden_layers"),
    )
    return Config(training_config, data_config, model_config)

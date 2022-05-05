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
    time_of_day: bool
    time_of_week: bool
    time_of_year: bool
    last_day_y: bool
    two_last_day_y: bool
    randomize_last_y: bool


@dataclass
class ModelConfig:
    lstm_depth: int
    hidden_layers: int

@dataclass
class Config:
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig


def read_config(path: str) -> Config:
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
        time_of_day = data.get("data").get("time_of_day", False),
        time_of_week = data.get("data").get("time_of_week", False),
        time_of_year = data.get("data").get("time_of_year", False),
        last_day_y = data.get("data").get("last_day_y", False),
        two_last_day_y = data.get("data").get("two_last_day_y", False),
        randomize_last_y = data.get("data").get("randomize_last_y", False)
    )
    model_config = ModelConfig(
        lstm_depth = data.get("model").get("lstm_depth"),
        hidden_layers = data.get("model").get("hidden_layers"),
    )
    return Config(training_config, data_config, model_config)

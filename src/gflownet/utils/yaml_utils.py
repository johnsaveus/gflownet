import yaml
from dataclasses import is_dataclass, fields
from typing import Type, Any
from gflownet.config import Config


def dict2cls(data_class: Type, data: dict) -> Any:
    """
    Recursively initialize a dataclass from a dictionary.
    """
    if not is_dataclass(data_class):
        raise TypeError(f"{data_class} is not a dataclass")

    # Initialize the dataclass fields from the dictionary
    field_values = {}
    for field in fields(data_class):
        field_name = field.name
        field_type = field.type
        # Check the common name in yaml and config
        if field_name in data:
            # Yaml is loaded as dict
            value = data[field_name]
            if is_dataclass(field_type):
                # Convert the dict to dataclass type
                field_values[field_name] = dict2cls(field_type, value)
            else:
                field_values[field_name] = value

    return data_class(**field_values)


def yml2cfg(file_path: str) -> Config:
    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)
    # Pass the dictionary as keyword arguments to the Config constructor
    return dict2cls(Config, config_dict)

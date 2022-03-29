## This file defines functions that are aliases of combinations of system functions, i.e. those that do not depend on functions defined in this module.
import os
import json
import pickle
import hashlib
from typing import Any, Dict, List, Union


full_path = lambda relative_path: os.path.realpath(os.path.expanduser(relative_path))

iterator_size = lambda iterator: sum(1 for _ in iterator)

def json_pretty_print(data_to_print: Union[Dict, List]):
    print(
        json.dumps(
            data_to_print,
            indent=4
        )
    )

def json_pretty_str(data_to_print: Union[Dict, List]):
    if type(data_to_print) is dict:
        data_to_print_json_serializable = {
            str(key): str(value) for key, value in data_to_print.items()
        }
    elif type(data_to_print) is list:
        data_to_print_json_serializable = [
            str(element) for element in data_to_print
        ]
    return json.dumps(
                data_to_print_json_serializable,
                indent=4
            )

def json_pretty_write(data_to_write: Union[Dict, List], output_path: str):
    with open(output_path, 'w') as data_writer:
        json.dump(data_to_write, data_writer, indent=4)


def json_load(json_data_path: str):
    with open(json_data_path, 'r') as json_data_reader:
        json_data = json.load(json_data_reader)
    return json_data


def pickle_dump(data_to_dump: Any, data_path: str):
    with open(data_path, 'wb') as data_dumper:
        pickle.dump(data_to_dump, data_dumper)


def pickle_load(data_path: str):
    with open(data_path, 'rb') as data_loader:
        pickle_data = pickle.load(data_loader)
    return pickle_data


def simple_hash(string: str):
    hash_value = hashlib.sha256(string.encode("utf-8")).hexdigest()
    return hash_value

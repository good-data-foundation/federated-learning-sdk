import yaml
import torch.optim as op
from torch.nn import modules

import goodDataML.learning.horizontal.torch_model.process_data as processes
from typing import Callable


def convert_loss_name_to_func(loss):
    # type: (str) -> modules.module
    return getattr(modules.loss, loss)


def convert_optimizer_to_func(optimizer):
    # type: (str) -> op.optimizer.Optimizer
    return getattr(op, optimizer)


def convert_data_process_to_func(process):
    # type: (str) -> Callable
    return getattr(processes, process)


def get_config_from_config_file(config_file: bytes):
    """
    Load parameters from the configuration file
    :param config_file
    :return dict
    """
    # open config file
    data_and_model_config = yaml.load(config_file)

    # replace string with function
    numeric_features_process_func \
        = data_and_model_config['data_config']['feature_process']['numeric_features']
    data_and_model_config['data_config']['feature_process']['numeric_features'] \
        = convert_data_process_to_func(numeric_features_process_func)

    categorical_features_process_func \
        = data_and_model_config['data_config']['feature_process']['categorical_features']
    data_and_model_config['data_config']['feature_process']['categorical_features'] \
        = convert_data_process_to_func(categorical_features_process_func)

    loss_fn_func = data_and_model_config['model_config']['model_train_config']['loss_fn']
    data_and_model_config['model_config']['model_train_config']['loss_fn'] \
        = convert_loss_name_to_func(loss_fn_func)

    optimizer_func = data_and_model_config['model_config']['model_train_config']['optimizer']
    data_and_model_config['model_config']['model_train_config']['optimizer'] \
        = convert_optimizer_to_func(optimizer_func)

    return data_and_model_config


def send_data_to_device(batch_data, device):
    # type: (list, str) -> list
    if device == 'cuda':
        batch_data = [data.to(device) for data in batch_data]
    else:
        pass
    return batch_data

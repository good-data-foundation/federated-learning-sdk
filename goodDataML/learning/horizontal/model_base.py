import numpy as np
from torch.nn import Module
from typing import Dict, List, Any


class BaseMmodel(object):

    def __init__(self):
        # type: () -> None
        self.model = Module()

    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        pass

    def load_model(self, model_dict):
        # type: (str) -> None
        pass

    def save_model(self, model_dict):
        # type: (str) -> None
        pass

    def mpc_update_param(self, data_size, epoch):
        # type: (int, int) -> None
        pass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from typing import Dict, List, Any


class PytorchDataset(Dataset):
    """
    Generate Pytorch data set for model
    """

    def __init__(self, df, config):
        # type: (pd.DataFrame, Dict[str, Any]) -> None
        """
        Currently assume the df is pandas data frame, but in the future it could be
        numpy array, or other format.

        :param df:
        :param config:
        """
        self.df = df
        self.data_config = config['data_config']
        self.target = self.data_config['feature_type']['label']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # type: (int) -> List[Any]

        num_feature = self._convert_data_to_tensor(idx, self.data_config['feature_type']['numeric_features'])
        cate_feature = self._convert_data_to_tensor(idx, self.data_config['feature_type']['categorical_features'])

        features = (num_feature, cate_feature)
        target = torch.LongTensor(self.df[self.target].values[idx]).squeeze()
        sample = [features, target]

        return sample

    def _convert_data_to_tensor(self, idx, features):
        # type: (int, str) -> torch.FloatTensor
        numeric_feature_list = torch.FloatTensor(np.vstack(self.df[features].iloc[idx].values))
        return numeric_feature_list


class PytorchIterDataset(IterableDataset):
    pass

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer


def normalization(data: np.array):
    """
    Maximum and minimum normalization of data
        data = (data - data_min) / (data_max - data_min)
    :param data
    :return data
    """
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    return data


def standardization(data: np.array):
    """
    Data standardization
    """
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    data = (data - mu) / sigma
    return data


def one_hot_encode(data: np.array):
    """
    one hot encode
    """
    return LabelBinarizer().fit_transform(data)


def process_data(df, config):
    # type: (pd.DateFrame, dict) -> pd.DateFrame
    data_config = config['data_config']
    if data_config['feature_process']['numeric_process']:
        num_process_func = data_config['feature_process']['numeric_process']
        for num_ftr in data_config['feature_type']['numeric_features']:
            df[num_ftr] = num_process_func(df[num_ftr].values)

    if data_config['feature_process']['categorical_features']:
        cate_process_func = data_config['feature_process']['categorical_features']
        pass

    return df


def get_features_data(df, data_config):
    """
    get and process data
    """

    # add numerical data

    data = df[data_config['feature_type']['numeric_features']].values
    data.astype(np.float64)

    # add categorical data
    for feature in data_config['feature_type']['categorical_features']:
        # get values
        categorical_feature_data = df[feature].values

        # string encode
        encode_categorical_feature_data \
            = data_config['feature_process']['categorical_features'](categorical_feature_data)
        # Converts the encoded string to the base type
        int_categorical_feature_data \
            = np.max(encode_categorical_feature_data,
                     axis=1) + np.argmax(encode_categorical_feature_data, axis=1)
        float_categorical_feature_data = int_categorical_feature_data.astype(np.float64)

        # transpose
        float_categorical_feature_data = float_categorical_feature_data.reshape(-1, 1)

        # matrix splicing
        data = np.hstack((data, float_categorical_feature_data))

    where_are_nan = np.isnan(data)
    data[where_are_nan] = 0

    # process data
    data = data_config['feature_process']['numeric_process'](data)

    return data

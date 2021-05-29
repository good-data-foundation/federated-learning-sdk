import time
import asyncio
import pandas as pd
from numpy import array
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import syft
from syft.workers.websocket_client import WebsocketClientWorker


class ArgClass(object):
    def __init__(self, ):
        self.train_size = 0.8
        self.batch_size = 2
        self.epochs = 50
        self.lr = 0.01
        self.path = r'..\data\train.csv'
        self.target = 'Survived'
        self.features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']


class Net(nn.Module):
    """
    test model
    """

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


def send_data_to_worker(data: array, features: array,
                        target: array, compute_nodes: list,
                        args: ArgClass):
    """
    send data to websocket compute node
    :param data: numpy.array
    :param features
    :param target
    :param compute_nodes
    :param args
    :return list
    """
    remote_dataset = []
    data_tensor = torch.tensor(data[features].values, requires_grad=False).float()
    target_tensor = torch.from_numpy(data[target].values).float()

    train = TensorDataset(data_tensor, target_tensor)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.send(compute_nodes)
        target = target.send(compute_nodes)
        remote_dataset.append((data, target))
    return remote_dataset


def update(data: syft.PointerTensor, target: syft.PointerTensor, model, optimizer):
    """
    train one batch data for DO
    :param data
    :param target
    :param model
        model is a custom model
    :param optimizer optimizer is a optimizer function in torch.optim lib
    :return updated model
    """
    model.send(data.location)
    optimizer.zero_grad()
    pred = model(data)
    loss = nn.BCELoss()
    output = loss(pred.squeeze(), target)
    output.backward()
    optimizer.step()
    return model


async def share_param(remote_index: int, params: list,
                      mpc_nodes: list, crypto_provider: WebsocketClientWorker):
    """
    async function, DO share param to MPC nodes
    :param remote_index: DO's location in list of DOs
    :param params
    :param mpc_nodes
    :param crypto_provider
    return
    """
    params_remote_index = list()
    for param_i in range(len(params[0])):
        params_remote_index.append(
            params[remote_index][param_i].fix_precision().get().share(
                *mpc_nodes,
                crypto_provider=crypto_provider
            ).get())
    return params_remote_index


def test(models: list, test_loader: DataLoader):
    """
    test the model
    :param models
    :param test_loader
    :param target: the target type is random and depends on the data being queried
    :return
    """
    models[0].eval()
    test_loss = 0
    for data, target in test_loader:
        output = models[0](data)
        loss = nn.BCELoss()
        test_loss += loss(output.squeeze(), target)

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))


def get_train_data_test_data_feature_target(args: ArgClass):
    """
    custom dataSet and features
    the "train.csv" can be download in https://www.kaggle.com/c/titanic/data?select=train.csv
    :return
        train_data: DataFrame or TextParser
        test_data: DataFrame or TextParser
        features: DataFrame or TextParser
        target: DataFrame or TextParser
    """
    features = args.features
    # features = args.numeric_features
    target = args.target
    train_data = pd.read_csv(args.path)
    train_data = train_data.fillna(0)
    t_size = int(args.train_size * len(train_data))
    test_data = train_data[t_size:]
    train_data = train_data[:t_size]

    return train_data, test_data, features, target


def get_remote_datasets(train_data, features, target, compute_nodes,
                        args: ArgClass):
    """
    Send data to remote computing node and get Pointer
    :param train_data: PySyft Pointer to point remote dataSet
    :param features: PySyft Pointer to point remote dataSet
    :param target
    :param compute_nodes: list
    :param args
    :return Pointer to point remote dataSet in websocket client
    """
    training_data_size = int(len(train_data) / len(compute_nodes))
    remote_datasets = []
    left = 0
    right = training_data_size
    for i in range(len(compute_nodes)):
        remote_datasets.append(send_data_to_worker(
            train_data[left:right], features, target, compute_nodes[i], args))
        left = right
        right += training_data_size
    return remote_datasets


async def train(compute_nodes: list,
                remote_dataset: list,
                models: list,
                optimizers: list,
                params: list,
                crypto_provider: WebsocketClientWorker,
                mpc_nodes: tuple):
    """
    Realize multi-party asynchronous federated learning
    :param compute_nodes
    :param remote_dataset
    :param models: Multi-party model
    :param optimizers
    :param params: model parameters
    :param crypto_provider
    :param mpc_nodes
    return models
    """

    for data_index in range(len(remote_dataset[0]) - 1):
        # we encrypt it on the remote machine
        pool = ThreadPoolExecutor(max_workers=10)
        tasks = []
        for remote_index in range(len(compute_nodes)):
            data, target = remote_dataset[remote_index][data_index]
            tasks.append(pool.submit(update,
                                     data,
                                     target,
                                     models[remote_index],
                                     optimizers[remote_index]))

        # wait all DOs to finish training
        wait(tasks, return_when=ALL_COMPLETED)
        for remote_index in range(len(compute_nodes)):
            models[remote_index] = tasks[remote_index].result()

        # # encrypted aggregation
        new_params = list()
        tasks = [asyncio.create_task(share_param(remote_index,
                                                 params,
                                                 mpc_nodes,
                                                 crypto_provider))
                 for remote_index in range(len(compute_nodes))]
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        new_params_list = list()
        for task in tasks:
            new_params_list.append(task.result())

        for param_i in range(len(params[0])):
            new_params.append(
                sum(new_params_list[remote_index][param_i]
                    for remote_index in range(len(compute_nodes))).
                float_precision() / len(compute_nodes)
            )

        # cleanup
        with torch.no_grad():
            for model_param in params:
                for param in model_param:
                    param = param.get()
                    param *= 0

            for remote_index in range(len(compute_nodes)):
                for param_index in range(len(params[remote_index])):
                    params[remote_index][param_index].set_(new_params[param_index])

    return models


def train_model(compute_nodes: list, crypto_provider: WebsocketClientWorker,
                model, mpc_nodes: tuple):
    """
    The main process of model training
    :param compute_nodes
    :param crypto_provider
    :param model
    :param mpc_nodes
    :return model: Trained model
    """

    args = ArgClass()
    # args = deal_base_data_config.define_and_get_arguments_from_config()

    # Get training data, test data, features, target
    train_data, test_data, features, target = get_train_data_test_data_feature_target(args)

    # Get remote training dataset
    remote_datasets = get_remote_datasets(train_data,
                                          features,
                                          target,
                                          compute_nodes,
                                          args)

    # Get test loader
    data_test = torch.from_numpy(test_data[features].values).float()
    target_test = torch.from_numpy(test_data[target].values).float()
    test_dataset = TensorDataset(data_test, target_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    models = []
    for i in range(len(compute_nodes)):
        models.append(model.copy())

    params = [list(compute_node_model.parameters()) for compute_node_model in models]
    optimizers = [optim.Adam(compute_node_model.parameters(), lr=args.lr)
                  for compute_node_model in models]

    # training and test
    print("start time: ", time.strftime("%X"))
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        asyncio.run(train(compute_nodes,
                          remote_datasets,
                          models,
                          optimizers,
                          params,
                          crypto_provider,
                          mpc_nodes))
        test(models, test_loader)

    print("end time: ", time.strftime("%X"))

    return models[0]

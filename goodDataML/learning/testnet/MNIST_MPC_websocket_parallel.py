import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

import syft
from syft.workers.websocket_client import WebsocketClientWorker
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

from goodDataML.learning.testnet.utils import get_do_uuid_from_info, get_mpc_uuid_from_info
from goodDataML.connection.chain import GoodDataChain, ChainEvent


class MNISTDataset(Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def update(data: syft.PointerTensor, target: syft.PointerTensor, model, optimizer):
    """
    train one batch data for DO
    :param data
    :param target
    :param model
    :param optimizer optimizer is a optimizer function in torch.optim lib
    :return updated model
    """
    model.send(data.location)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return model


async def share_param(remote_index: int, params: list,
                      mpc_nodes: tuple):
    """
    async function, DO share param to MPC nodes
    :param remote_index: DO's location in list of DOs
    :param params
    :param mpc_nodes
    :return
    """
    params_remote_index = list()
    for param_i in range(len(params[0])):
        params_remote_index.append(
            params[remote_index][param_i].fix_precision().get().share(
                *mpc_nodes,
                crypto_provider=None
            ).get())
    return params_remote_index


def predict(chain: GoodDataChain,
            prediction_uuid: str,
            model):
    """
    test the trained model
    :param chain
    :param prediction_uuid
    :param model
    :return
    """
    # set logs
    prediction_logs = dict()
    prediction_logs["start_time"] = int(time.time())

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    kwargs = {'batch_size': 64}
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nPredict set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    prediction_logs["loss"] = test_loss
    prediction_logs["accuracy"] = 1. * correct / len(test_loader.dataset)
    prediction_logs["end_time"] = int(time.time())
    chain.add_log(prediction_uuid, prediction_logs)


def validate(chain: GoodDataChain,
             epoch: int,
             total_epochs: int,
             query_uuid: str,
             query_info_and_nodes_info,
             models: list,
             test_loader: torch.utils.data.DataLoader):
    """
    validate the model
    :param chain
    :param epoch
    :param total_epochs
    :param query_uuid
    :param query_info_and_nodes_info
    :param models
    :param test_loader
    :return
    """
    # set logs
    test_logs = dict()
    test_logs["task"] = "validation"
    test_logs["epoch"] = epoch + 1
    test_logs["total_epochs"] = total_epochs
    test_logs["start_time"] = int(time.time())

    models[0].eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = models[0](data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_logs["loss"] = test_loss
    test_logs["accuracy"] = 1. * correct / len(test_loader.dataset)
    test_logs["end_time"] = int(time.time())
    chain.add_log(query_uuid, test_logs)


def define_and_get_arguments():
    """
    Define the hyperparameters required, features and target for training
    :return dic
    """
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size of the training")
    parser.add_argument("--test_batch_size", type=int, default=1000, help="batch size of test")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=1.0, help="learning rate")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--target", type=str, default='Survived', help="training target")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websockets client workers will " "be started in verbose mode",
    )
    args = parser.parse_args(args=[])
    return args


def send_data_to_worker(data: np.array,
                        target: np.array,
                        transform: transforms,
                        compute_node: WebsocketClientWorker,
                        batch_size: int):
    """
    send data to websocket compute node
    :param data: numpy.array
    :param target
    :param transform
    :param compute_node
    :param batch_size
    :return list
    """
    remote_dataset = []
    train_dataset = MNISTDataset(data, target, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.send(compute_node)
        target = target.send(compute_node)
        remote_dataset.append((data, target))
    return remote_dataset


def get_remote_dataset_and_validation_dataloader(batch_size: int, compute_nodes: list, data_set: int):
    """
    1) get training data and send to remote DO
    2) get test DataLoader
    :param batch_size:
    :param compute_nodes:
    :param data_set:
    :return
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    kwargs = {'batch_size': batch_size}

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)

    # get remote datasets
    print("data_set = ", data_set)
    if data_set == 1:
        size = 1000
    else:
        size = 10000
    print("dataset size: ", size)
    data = dataset1.data[:size]
    targets = dataset1.targets[:size]
    training_data_size = int(len(data) / len(compute_nodes))
    remote_datasets = []
    left = 0
    right = training_data_size
    for i in range(len(compute_nodes)):
        remote_datasets.append(send_data_to_worker(
            data[left:right], targets[left:right], transform, compute_nodes[i], batch_size))
        left = right
        right += training_data_size

    # test Dataloader
    validation_data = dataset1.data[50000:]
    validation_target = dataset1.targets[50000:]
    validation_dataset = MNISTDataset(validation_data, validation_target, transform)
    test_loader = torch.utils.data.DataLoader(validation_dataset, **kwargs)

    return remote_datasets, test_loader


async def train(chain: GoodDataChain,
                epoch: int,
                total_epochs: int,
                query_uuid: str,
                query_info_and_nodes_info: dict,
                compute_nodes: list,
                remote_dataset: list,
                models: list,
                optimizers: list,
                params: list,
                mpc_nodes: tuple):
    """
    Realize multi-party asynchronous federated learning
    :param chain: write a journal to the chain
    :param epoch
    :param total_epochs
    :param query_uuid
    :param query_info_and_nodes_info
    :param compute_nodes
    :param remote_dataset
    :param models: Multi-party model
    :param optimizers
    :param params: model parameters
    :param mpc_nodes
    :return models
    """

    do_uuids = get_do_uuid_from_info(query_info_and_nodes_info)
    mpc_uuids = get_mpc_uuid_from_info(query_info_and_nodes_info)
    for data_index in range(len(remote_dataset[0])):
        print("batch_index: ", data_index)
        # we encrypt it on the remote machine
        pool = ThreadPoolExecutor(max_workers=10)
        tasks = []
        train_logs = dict()
        train_logs["task"] = "train"
        train_logs["epoch"] = epoch
        train_logs["total_epochs"] = total_epochs
        train_logs["batch_index"] = data_index
        train_logs["start_time"] = int(time.time())
        train_logs["do_uuid"] = do_uuids
        train_logs["mpc_uuid"] = mpc_uuids

        for remote_index in range(len(compute_nodes)):
            data, target = remote_dataset[remote_index][data_index]
            tasks.append(pool.submit(update, data, target,
                                     models[remote_index], optimizers[remote_index]))

        # wait all DOs to finish training
        wait(tasks, return_when=ALL_COMPLETED)
        for remote_index in range(len(compute_nodes)):
            models[remote_index] = tasks[remote_index].result()

        # encrypted aggregation
        new_params = list()
        tasks = [asyncio.create_task(share_param(remote_index,
                                                 params,
                                                 mpc_nodes,
                                                 ))
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
        train_logs["total_batch_indexes"] = len(remote_dataset[0])
        train_logs["end_time"] = int(time.time())
        # add_logs_to_chain(chain, query_uuid, train_logs)
        chain.add_log(query_uuid, train_logs)
        # clean up
        with torch.no_grad():
            for model_param in params:
                for param in model_param:
                    param = param.get()
                    param *= 0

            for remote_index in range(len(compute_nodes)):
                for param_index in range(len(params[remote_index])):
                    params[remote_index][param_index].set_(new_params[param_index])

    return models


def train_model(chain: GoodDataChain,
                query_uuid: str,
                query_info_and_nodes_info: dict,
                compute_nodes: list,
                model,
                mpc_nodes: tuple):
    """
    The main process of model training
    :param chain
    :param query_uuid
    :param query_info_and_nodes_info
    :param compute_nodes
    :param model
    :param mpc_nodes
    :return model: Trained model
    """
    print("start training.")
    args = define_and_get_arguments()
    remote_datasets, test_loader = get_remote_dataset_and_validation_dataloader(
                                        args.batch_size,
                                        compute_nodes,
                                        query_info_and_nodes_info.query_info.data_set
                                    )

    models = []
    for i in range(len(compute_nodes)):
        models.append(model.copy())

    params = [list(compute_node_model.parameters()) for compute_node_model in models]
    optimizers = [optim.Adadelta(compute_node_model.parameters(), lr=args.lr)
                  for compute_node_model in models]

    # training and test
    print("start time: ", time.strftime("%X"))

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        asyncio.run(train(chain,
                          epoch,
                          args.epochs,
                          query_uuid,
                          query_info_and_nodes_info,
                          compute_nodes,
                          remote_datasets,
                          models,
                          optimizers,
                          params,
                          mpc_nodes))
        validate(chain, epoch, args.epochs, query_uuid, query_info_and_nodes_info, models, test_loader)

    print("end time: ", time.strftime("%X"))

    # Close websocket connection
    for compute_node in compute_nodes:
        if isinstance(compute_node, WebsocketClientWorker):
            compute_node.close()

    for mpc_node in mpc_nodes:
        if isinstance(mpc_node, WebsocketClientWorker):
            mpc_node.close()

    return models[0]

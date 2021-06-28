import json

import torch
from syft.workers.websocket_client import WebsocketClientWorker

from goodDataML.learning.horizontal.torch_model.testnet_models import Net1, Net2


def get_nodes_from_query_info(query_and_running_info, hook):
    """
    get nodes info from query response and return nodes
    :param query_and_running_info:
    :param hook:
    :return: data_owners, encrypt_provider, mpc_nodes
    """
    data_owners_info = query_and_running_info.query_cluster.do_nodes
    mpc_nodes_info = query_and_running_info.query_cluster.mpc_nodes
    data_owners = []
    mpc_nodes = []
    for i in range(len(data_owners_info)):
        data_owners.append(WebsocketClientWorker(id="worker" + str(i + 1),
                                                 port=int(data_owners_info[i].port),
                                                 host=data_owners_info[i].address,
                                                 hook=hook))

    for i in range(len(mpc_nodes_info)):
        mpc_nodes.append(WebsocketClientWorker(id="mpc" + str(i + 1),
                                               port=int(mpc_nodes_info[i].port),
                                               host=mpc_nodes_info[i].address,
                                               hook=hook))

    return data_owners, mpc_nodes


def load_model_from_path(path: str):
    """
    load model from path
    :param path: str
    :return model
    """
    model = torch.load(path)

    return model


def save_model_to_path(model, path: str):
    """
    Save model to path
    :param path: str
    :return model
    """
    torch.save(model, path)


def get_do_uuid_from_info(query_info_and_nodes_info):
    do_uuids = list()
    for do_node in query_info_and_nodes_info.query_cluster.do_nodes:
        do_uuids.append(do_node.uuid)
    return do_uuids


def get_mpc_uuid_from_info(query_info_and_nodes_info):
    mpc_uuids = list()
    for mpc_node in query_info_and_nodes_info.query_cluster.mpc_nodes:
        mpc_uuids.append(mpc_node.uuid)
    return mpc_uuids


def get_config_file(filename):
    with open(filename) as fr:
        config = json.load(fr)
    if config is None:
        raise RuntimeError("Cannot open config file.")
    return config

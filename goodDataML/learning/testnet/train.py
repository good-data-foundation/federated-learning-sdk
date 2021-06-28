import os
import time
from binascii import unhexlify

import grpc
import syft

from goodDataML.learning.testnet.utils import *
from goodDataML.connection.gooddata import get_query_info, query_completed
from goodDataML.learning.testnet.MNIST_MPC_websocket_parallel import train_model
from goodDataML.learning.horizontal.torch_model.testnet_models import Net1, Net2


def training_workflow(chain, event):
    """
    callback function
    :param chain: the chain that you monitor
    :param event: dict, includes some query request information submitted
    :returns: None
    """
    hook = syft.TorchHook(torch)
    trained_model_path = './tmp/model_uuid_completed'
    config = get_config_file('./local_config.json')
    channel = grpc.insecure_channel(config['good_data_service_address'],
                                    options=[
                                        ('grpc.max_send_message_length', 5 * 1024 * 1024),
                                        ('grpc.max_receive_message_length',
                                         5 * 1024 * 1024),
                                    ])

    print("The listener event was successfully retrieved")
    bytes_str = unhexlify(event['queryUUID'][2:])
    query_uuid = bytes_str.decode(encoding='utf-8')
    print("query_uuid: ", query_uuid)

    # set start time of deal query request
    query_execution_info_dict = {'_uuid': query_uuid,
                                 'start_time_ns': int(time.time()),
                                 'finished_time_ns': -1,
                                 'status': 1}

    # get query info from GDS
    query_info_and_nodes_info, model_path = get_query_info(query_uuid, channel)

    print("query_info_and_nodes_info.query_info.data_set: ", query_info_and_nodes_info.query_info.data_set)
    # get nodes info
    compute_nodes, mpc_nodes = \
        get_nodes_from_query_info(query_info_and_nodes_info, hook)

    do_uuids = get_do_uuid_from_info(query_info_and_nodes_info)

    # load model
    print("model_path: ", model_path)
    model = load_model_from_path(model_path)

    # train model and get trained model
    model_trained = train_model(chain,
                                query_uuid,
                                query_info_and_nodes_info,
                                compute_nodes,
                                model,
                                mpc_nodes)

    # set model_completed path
    filename_completed = trained_model_path + str(time.time()) + '.pt'
    save_model_to_path(model_trained, filename_completed)

    # query complete
    query_execution_info_dict['finished_time_ns'] = int(time.time())
    model_completed_path = os.path.abspath(filename_completed)

    for do_uuid in do_uuids:
        query_completed(do_uuid, model_completed_path,
                        channel, query_execution_info_dict)

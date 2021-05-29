from binascii import unhexlify

import grpc

from goodDataML.learning.testnet.utils import *
from goodDataML.connection.gooddata import get_query_execution_info, test_model_completed
from goodDataML.learning.testnet.MNIST_MPC_websocket_parallel import predict
from goodDataML.learning.horizontal.torch_model.testnet_models import Net1, Net2


def prediction_workflow(chain, event):
    """
    callback function
    :param chain: the chain that you monitor
    :param event: dict, includes some query request information submitted
    :return
    """
    print("The prediction event was successfully retrieved")
    config = get_config_file('./local_config.json')
    channel = grpc.insecure_channel(config['good_data_service_address'],
                                    options=[
                                        ('grpc.max_send_message_length', 5 * 1024 * 1024),
                                        ('grpc.max_receive_message_length',
                                         5 * 1024 * 1024),
                                    ])

    bytes_str = unhexlify(event['queryUUID'][2:])
    query_uuid = bytes_str.decode(encoding='utf-8')

    bytes_str = unhexlify(event['predictionUUID'][2:])
    prediction_uuid = bytes_str.decode(encoding='utf-8')

    # get query info from GDS
    prediction_info_and_nodes_info, model_path = get_query_execution_info(query_uuid, channel)

    # load model
    print("model_path: ", model_path)
    model = load_model_from_path(model_path)

    # test model
    predict(chain, prediction_uuid, model)

    test_model_completed(prediction_uuid, channel)

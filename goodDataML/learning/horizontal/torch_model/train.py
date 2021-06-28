import os
import time
import pandas as pd
from grpc import Channel
from binascii import unhexlify
from torch.utils.data import DataLoader

from goodDataML.mpc.mpc_status import MPCServerStatus
from goodDataML.connection.gooddata import get_query_info, query_completed

from goodDataML.learning.horizontal.torch_model.model_class import PytorchModel
from goodDataML.learning.horizontal.torch_model.process_data import process_data
from goodDataML.learning.horizontal.torch_model.data_class import PytorchDataset
from goodDataML.learning.utils.pytorch_utils import get_config_from_config_file

from typing import Dict


def training(df, do_uuid, event, channel):
    # type: (pd.DataFrame, str, Dict[str, str], Channel) -> None

    # get query uuid
    bytes_str = unhexlify(event['queryUUID'][2:])
    query_uuid = bytes_str.decode(encoding='utf-8')

    # set start time of deal query request
    query_execution_info_dict = {'_uuid': query_uuid,
                                 'start_time_ns': int(time.time()),
                                 'finished_time_ns': 1,
                                 'status': 1}

    # get query info from GDS
    query_info_and_nodes_info, model_path, config = get_query_info(query_uuid, channel)
    config = get_config_from_config_file(config)

    # connect MPC
    server_status = MPCServerStatus()

    # set model class
    model = PytorchModel(server_status, query_uuid, config)

    # load model
    model.load_model(model_path)

    # process_data
    df = process_data(df, config)

    # build data loader
    data_set = PytorchDataset(df, config)
    data_loader = DataLoader(data_set, batch_size=config['data_config']['batch_size'])

    # train model and get trained model
    model.fit(data_loader)

    # set model_completed path
    filename_completed = './model_uuid_completed' + str(time.time()) + '.pt'
    model.save_model(filename_completed)

    # query complete
    query_execution_info_dict['finished_time_ns'] = int(time.time())
    model_completed_path = os.path.abspath(filename_completed)
    query_completed(do_uuid, model_completed_path, channel, **query_execution_info_dict)

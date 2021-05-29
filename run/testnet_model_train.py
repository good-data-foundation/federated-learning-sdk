import sys
import os
import threading
import time
import queue
import pickle

import grpc

from goodDataML.learning.testnet.utils import get_config_file

sys.path.append('../')
from goodDataML.connection.chain import GoodDataChain, ChainEvent
from goodDataML.learning.testnet.train import training_workflow
from goodDataML.learning.testnet.prediction import prediction_workflow
from goodDataML.learning.horizontal.torch_model.testnet_models import Net1, Net2


def get_queue():
    if os.path.exists('./training_task_queue'):
        mutex.acquire()
        with open('./training_task_queue', 'rb') as file:
            train_event_queue = pickle.load(file)
        mutex.release()
    else:
        mutex.acquire()
        train_event_queue = []
        with open('./training_task_queue', 'wb') as file:
            pickle.dump(train_event_queue, file)
        mutex.release()
    return train_event_queue


def train_event(chain, event):
    print("call back train.")
    mutex.acquire()
    train_event_queue.append({"event": event})
    with open('./training_task_queue', 'wb') as file:
        pickle.dump(train_event_queue, file)
    mutex.release()


if __name__ == '__main__':
    """
    The boot function of the training script.
    
    steps:
    0) Start register_DOs.py by "python register_DOs.py" (run only once)
    1) Start start_websocket_servers.py by "python start_websocket_servers.py"
    """
    config = get_config_file('./local_config.json')

    mutex = threading.Lock()
    train_event_queue = get_queue()
    # monitor the chain and set the callback function

    chain = GoodDataChain(config['chain_address'],
                          config['private_key'],
                          {})

    chain.subscribe(ChainEvent.QuerySubmitted, train_event)
    chain.subscribe(ChainEvent.PredictionSubmitted, prediction_workflow)
    print(time.strftime("%X"), " Script is starting")

    while True:
        if train_event_queue:
            train_event = train_event_queue[0]
            print("get event: {} from train event.".format(train_event))
            training_workflow(chain, train_event["event"])
            mutex.acquire()
            train_event_queue.pop(0)
            with open('./training_task_queue', 'wb') as file:
                pickle.dump(train_event_queue, file)
            mutex.release()
        time.sleep(1)

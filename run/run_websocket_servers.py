from multiprocessing import Process
import argparse
import os

import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker
import torch

from goodDataML.learning.testnet.utils import get_config_file

hook = sy.TorchHook(torch)


def start_proc(participant, kwargs):  # pragma: no cover
    """
    helper function for spinning up a websocket participant
    :param participant:
    :param kwargs:
    :return:
    """

    def target():
        server = participant(**kwargs)
        server.start()

    p = Process(target=target)
    p.start()
    return p


config = get_config_file(filename='./local_config.json')

parser = argparse.ArgumentParser(description="Run websocket server worker.")

parser.add_argument(
    "--port", "-p", type=int, help="port number of the websocket server worker, e.g. --port 8777"
)

parser.add_argument("--host", type=str, default=config['do1']['ip_address'], help="host for the connection")

parser.add_argument(
    "--id", type=str, help="name (id) of the websocket server worker, e.g. --id alice"
)

parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="if set, websocket server worker will be started in verbose mode",
)

args = parser.parse_args()

kwargs = {
    "id": args.id,
    "host": args.host,
    "port": args.port,
    "hook": hook,
    "verbose": args.verbose,
}

if os.name != "nt":
    server = start_proc(WebsocketServerWorker, kwargs)
else:
    server = WebsocketServerWorker(**kwargs)
    server.start()

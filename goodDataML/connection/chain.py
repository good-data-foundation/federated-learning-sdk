"""
Communicate with chain
"""
import json
from enum import Enum, unique
from goodDataML.connection.chain_connection_helpers import SmartContract, W3Helper,\
    ThreadManager, CHAIN_MAX_TASK_NUM, CHAIN_TASK_NAME_PREFIX, QUEUE_EXIT_SIGNAL
import logging

# const
EVENT_DEFAULT_QUEUE_SIZE = 10
CHAIN_CONTRACT_ADDRESS = '0x2222222222222222222222222222222222222222'


@unique
class ChainEvent(Enum):
    # name -> event name
    # value or signature -> event signature
    LogAdded = 'LogAdded(bytes uuid, bytes payload)'
    QuerySubmitted = 'QuerySubmitted(bytes queryUUID, bytes qcUUID, int64 dataSet)'
    PredictionSubmitted = 'PredictionSubmitted(bytes queryUUID, bytes predictionUUID)'

    @property
    def signature(self):
        return self._value_


class GoodDataSmartContract(SmartContract):
    """SorterOne Smart Contract"""

    def __init__(self, url: str, contract_addr: str, queue_size: int):
        """
        :param url: like this: 'http://localhost:8545' - websocket has some concurrency problem
        :param contract_addr: like this: '0x2222222222222222222222222222222222222222'
        :param queue_size:
        """
        super().__init__(W3Helper(url), contract_addr, queue_size)
        # smart contract function list
        self.attrs = {
            # There are no white space for function signature.
            # event signature can be:
            #     EventName(type1 name1, type2 name2)
            #     EventName(type1 indexed name1, type2 name2)
            #     EventName(type1, type2)
            #
            # wrong example:
            # "addLog": ("addLog(bytes,bytes)", False, ["LogAdded(bytes, bytes)"])
            # correct example:
            # "addLog": ("addLog(bytes,bytes)", False, ["LogAdded(bytes,bytes)"])

            # "function_name": (
            #   "function_signature",
            #   `True` if you want to add extra data to end of txdata,
            #   ["event_signature1", "event_signature2"],
            # )
            "addLog": (
                "addLog(bytes,bytes)",
                False,
                [ChainEvent.LogAdded.signature],
            ),
            "submitQuery": (
                "submitQuery(bytes,bytes,int64,uint256)",
                False,
                [ChainEvent.QuerySubmitted.signature],
            ),
            "submitPrediction": (
                "submitPrediction(bytes,bytes)",
                False,
                [ChainEvent.PredictionSubmitted.signature],
            )
        }


# GoodDataChain
class GoodDataChain:
    """
    GoodDataChain represent a role which could communicate with GoodData chain
    """

    def __init__(self, rpc_url: str, private_key: str, context: dict, queue_size=EVENT_DEFAULT_QUEUE_SIZE):
        """
        :param rpc_url: like this: 'ws://localhost:8546'
        :param private_key: like this:
            '872708f77fcdaa388ef3abdc99e331f3cd708d31007dd0bef89d4378c6bba0ac'
        :param context: All information about config are here. like this:
            # {
            #   "config1": {
            #       "ip": "127.0.0.1",
            #       "port": "80",
            #   },
            #   "config2": {
            #       "ip": "127.0.0.1",
            #       "port": "8080",
            #   },
            # }
        """
        self.log = logging.getLogger(self.__class__.__name__)
        self.private_key = private_key
        self.contract = GoodDataSmartContract(rpc_url, CHAIN_CONTRACT_ADDRESS, queue_size)
        self._thread_manager = ThreadManager(
            max_workers=CHAIN_MAX_TASK_NUM,
            thread_name_prefix=CHAIN_TASK_NAME_PREFIX)
        self.context = context

        # _subscriber: {
        #   event_name1: [callback1, callback2...],
        #   event_name2: [callback3, callback4...],
        # }
        self._subscriber = {}

        def _dispatcher():
            # create a thread to post event
            # to all subscriber who subscribe it
            while not self._thread_manager.has_shutdown():
                event = self.contract.events_queue.get()
                if event is QUEUE_EXIT_SIGNAL:
                    self.log.debug('Got queue exit signal')
                    # put QUEUE_EXIT_SIGNAL again
                    # help the queue at other thread could capture QUEUE_EXIT_SIGNAL
                    self.contract.events_queue.put(event)
                    break
                self.log.debug("Got event from queue %s", event)
                if event['event_name'] in self._subscriber:
                    for callback in self._subscriber[event['event_name']]:
                        self._thread_manager.submit(callback, self, event)

        self._thread_manager.submit(_dispatcher)

    def subscribe(self, event: ChainEvent, *callback):
        """
        subscribe event and assign list of callback
        all callback will be triggered if event occur
        :param event:
        :param callback:
        :return:
        """
        if event.name not in self._subscriber:
            self._subscriber[event.name] = list(callback)
            self._thread_manager.submit(
                self.contract.watch, event.signature, None)
        else:
            self._subscriber[event.name].append(callback)

    def shutdown(self, wait=False):
        """
        shutdown stop watching and close all threads
        :return:
        """
        self.contract.exit_all_watch(wait)
        self._thread_manager.shutdown(wait)

    def add_log(self, uuid: str, payload: dict):
        """
        SDK side needs a function to report status for Smart Contract.
        :param uuid: It is the id of the query.
        :param payload: It contains information about training.
        :return receipt: receipt is a dict, which include block and the event information
        """
        uuid_bytes = uuid.encode(encoding='utf-8')
        payload_bytes = json.dumps(payload).encode(encoding='utf-8')
        return self.contract.addLog(self.private_key, uuid_bytes, payload_bytes)

    def submit_query(self, query_uuid: bytes, qc_uuid: bytes, data_set: int, fees: int):
        """
        :param query_uuid:
        :param qc_uuid:
        :param data_set: select data set for train
        :param fees:
        :return:
        """
        return self.contract.submitQuery(self.private_key, query_uuid, qc_uuid, data_set, fees)

    def submit_prediction(self, query_uuid: bytes, prediction_uuid: bytes):
        """
        :param query_uuid:
        :param prediction_uuid:
        :return:
        """
        return self.contract.submitPrediction(self.private_key, query_uuid, prediction_uuid)

"""block chain and smart contract helpers"""

import logging
import time
from binascii import hexlify, unhexlify
import eth_utils
import eth_abi
from web3 import Web3
import queue
from concurrent.futures import ThreadPoolExecutor

# const
CHAIN_FUNC_CALL_TIMEOUT_IN_S = 120
CHAIN_EVENT_GET_FAILED_WAIT_TIME_IN_S = 5
CHAIN_MAX_TASK_NUM = 30
CHAIN_MAX_FUNC_CALL_NUM = 20
CHAIN_TASK_NAME_PREFIX = 'chain_task'
CONTRACT_MAX_TASK_NUM = 30
CONTRACT_TASK_NAME_PREFIX = 'contract_task'
QUEUE_EXIT_SIGNAL = object()


class ThreadManager:
    """
    ThreadManager control thread safe exit and max thread number
    """
    def __init__(self, _queue: queue.Queue = None, max_workers=CHAIN_MAX_TASK_NUM,
                 thread_name_prefix=CHAIN_TASK_NAME_PREFIX):
        self._pool = ThreadPoolExecutor(max_workers, thread_name_prefix=thread_name_prefix)
        self._shutdown = False
        self._queue = _queue

    def submit(self, func, *args, **kwargs):
        """
        create a new thread to run `func`
        :param func:
        :param args:
        :param kwargs:
        :return:
        """
        self._pool.submit(func, *args, **kwargs)

    def shutdown(self, wait=False):
        """
        shutdown notify thread quit
        :param wait: Whether to wait to release all resources
        :return:
        """
        if self._queue is not None:
            self._queue.put(QUEUE_EXIT_SIGNAL)
        self._shutdown = True
        self._pool.shutdown(wait)

    def has_shutdown(self):
        """
        this is a judge condition for `while not` instead of `while True`
        ensure thread could exit when has_shutdown() is True
        :return:
        """
        return self._shutdown


class W3Helper:
    """Web3 helper"""

    def __init__(self, endpoint: str):
        """
        :param endpoint: example: 'http://localhost:8545' or 'ws://localhost:8546'
        """

        # __nonce maintain correct number of transactions,
        # avoid transaction of same nonce in pending state could not execution.
        self.__nonce = {}
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            self.web3 = Web3(Web3.HTTPProvider(endpoint))
        elif endpoint.startswith("ws://") or endpoint.startswith("wss://"):
            self.web3 = Web3(Web3.WebsocketProvider(endpoint))
        else:
            raise Exception("Unsupported protocol")

    def __getattr__(self, name):
        """
        :param name:
        :return:
        """
        if hasattr(self.web3, name):
            return getattr(self.web3, name)
        # eth api in web3 is the most commonly used
        eth = getattr(self.web3, 'eth')
        if hasattr(eth, name):
            return getattr(eth, name)
        raise AttributeError

    def call_api(self, method, *args):
        """
        :param method:
        :param args:
        :return:
        """
        return self.web3.manager.request_blocking(method, list(args))

    def eth_call(self, transaction: dict):
        """
        :param transaction:
        :return:
        """
        return self.eth.call(transaction)

    def sign_transaction(self, src_private_key: str, transaction: dict):
        """
        sign transaction with src_private_key
        :param src_private_key:
        :param transaction:
        :return:
        """
        return self.eth.account.signTransaction(transaction, src_private_key)

    def send_raw_transaction(self, raw_transaction):
        """
        :param raw_transaction:
        :return:
        """
        return self.eth.sendRawTransaction(raw_transaction)

    def execute_transaction(self, src_private_key: str, transaction: dict):
        """
        sign transaction and send it's raw transaction
        :param src_private_key:
        :param transaction:
        :return:
        """
        signed_txn = self.sign_transaction(src_private_key, transaction)
        return self.send_raw_transaction(signed_txn.rawTransaction)

    def execute_and_wait_for_transaction(self, src_private_key: str, transaction: dict,
                                         timeout=CHAIN_FUNC_CALL_TIMEOUT_IN_S):
        """
        :param src_private_key:
        :param transaction:
        :param timeout:
        :return:
        """
        tx_hash = self.execute_transaction(src_private_key, transaction)
        return self.wait_receipt_for_transaction(tx_hash, timeout)

    def wait_receipt_for_transaction(self, tx_hash: str, timeout=CHAIN_FUNC_CALL_TIMEOUT_IN_S):
        """
        :param tx_hash:
        :param timeout:
        :return:
        """
        receipt = self.eth.waitForTransactionReceipt(tx_hash, timeout)
        if receipt is None:
            raise TimeoutError("Transaction %r timed out" % hexlify(tx_hash))
        while not receipt['blockNumber'] and timeout > 0:
            time.sleep(2)
            timeout -= 1
            receipt = self.eth.waitForTransactionReceipt(tx_hash, timeout)
        return receipt

    def get_nonce_for_next_transaction(self, acc, block_identifier='latest'):
        """
        :param acc: account
        :param block_identifier:
        :return:
        """
        if hasattr(self.__nonce, acc) is False:
            self.__nonce[acc] = self.eth.getTransactionCount(acc, block_identifier) - 1
        self.__nonce[acc] += 1
        return self.__nonce[acc]
        # return self.eth.getTransactionCount(acc, block_identifier)


class SmartContractFuncCall:
    """Smart Contract Function Call"""

    def __init__(self, w3h: W3Helper, contract_addr: str, attr: dict):
        """
        initialize necessary environment for smart contract function call
        :param w3h:
        :param contract_addr:
        :param attr:
        """
        self.log = logging.getLogger(SmartContractFuncCall.__name__)
        self.w3h = w3h
        self.contract_addr = contract_addr

        self.ret_type = None
        self.need_data = False
        self.events = []
        if isinstance(attr, tuple):
            self.signature = attr[0]
            self.need_data = attr[1]
            for event in attr[2]:
                self.events.append(SmartContractFuncCall.parse_event_signature(event))
        else:
            self.signature = attr

        idx = self.signature.find(")")
        if idx > 0:
            self.ret_type = self.signature[idx+1:]
            self.signature = self.signature[:idx+1]
        else:
            raise Exception('function signature is wrong')

    # pylint: disable=unsubscriptable-object
    def __call__(self, *args, **kwargs):
        """
        smart contract function call
        perform execute_call if no return value
        perform view_call if have return value
        :param args:
        :param kwargs:
        :return:
        """
        private_key = None
        extra_data = None
        if not self.ret_type:
            private_key = args[0]
            args = args[1:]
        if self.need_data:
            extra_data = args[-1]
            args = args[:-1]

        data = ''
        if args:
            data = SmartContractFuncCall.encode_funcall(self.signature, *args)
        else:
            data = SmartContractFuncCall.encode_funcall(self.signature)
        if extra_data:
            data = "%s%s" % (data, extra_data[2:])
        if private_key:
            receipt = self.execute_call(private_key, data, **kwargs)
            receipt = dict(receipt)
            if self.events:
                receipt['events'] = SmartContractFuncCall.decode_logs(self.events, receipt['logs'])
            return receipt
        try:
            ret_data = self.view_call(data)
            ret = eth_abi.decode_abi([self.ret_type], ret_data)[0]
            if self.ret_type == "string":
                return ret.decode()
            return ret
        except Exception:
            return None

    def view_call(self, data):
        """
        Call readonly method of smart contract
        without private key and spend balance
        :param data:
        :return:
        """
        tx = {
            "value": 0,
            "to": self.contract_addr,
            "data": data,
            "from": self.contract_addr,
            "gasPrice": self.w3h.web3.eth.gasPrice
        }
        gas = self.w3h.web3.eth.estimateGas(tx)
        tx["gas"] = gas
        return self.w3h.eth_call(tx)

    def execute_call(self, private_key: str, _data, max_recall_count=CHAIN_MAX_FUNC_CALL_NUM, **kwargs):
        """
        to call function of smart contract with private key and wait for transaction.
        calling smart contract with private key which is necessary.
        private key will unlock balance of account to pay fees about calling smart contract.
        :param private_key:
        :param _data:
        :param max_recall_count: a number of max recall this function.
        :param kwargs: default timeout is 60
        :return:
        """
        if max_recall_count <= 0:
            raise ValueError('max_recall_count is', max_recall_count)
        addr = self.w3h.web3.eth.account.privateKeyToAccount(private_key).address
        tx = {
            "value": 0,
            "to": self.contract_addr,
            "data": _data,
            "from": addr,
            "gasPrice": self.w3h.web3.eth.gasPrice
        }
        gas = self.w3h.web3.eth.estimateGas(tx)
        tx["gas"] = gas
        tx['nonce'] = self.w3h.get_nonce_for_next_transaction(addr)
        timeout = kwargs.get("timeout", 60)
        #print('###############tx:', tx)
        # return self.w3h.execute_and_wait_for_transaction(private_key, tx, timeout=timeout)
        try:
            receipt = self.w3h.execute_and_wait_for_transaction(private_key, tx, timeout=timeout)
            #print('transaction was submitted:', receipt)
            return receipt
        except ValueError:
            #print('recall', self.execute_call)
            time.sleep(1)
            return self.execute_call(private_key, _data, **kwargs, max_recall_count=max_recall_count-1)

    @staticmethod
    def encode_funcall(func_type, *args):
        """
        Encode for contract method call
        :param func_type:
        :param args:
        :return:
        """
        func_type = func_type.replace(' ', '')
        signature = hexlify(eth_utils.keccak(func_type.encode()))[:8].decode()
        params = []
        values = list(args)
        idx = func_type.find("(")
        parmstr = func_type[idx+1:-1]
        data = ''
        if parmstr:
            for parm in func_type[idx+1:-1].split(","):
                params.append(parm.strip())
            data = hexlify(eth_abi.encode_abi(params, values)).decode()
        return "0x%s%s" % (signature, data)

    @staticmethod
    def parse_event_signature(event_signature: str):
        """parse event signatuer to dictionary"""
        event_signature = event_signature.strip()
        lidx = event_signature.find('(')
        ridx = event_signature.find(')')
        event_name = event_signature[:lidx]

        params_str = event_signature[lidx + 1:ridx]
        params = params_str.split(",")

        parameters = []
        for i in range(len(params)):
            param_tokens = params[i].strip().split()
            param = {'indexed': False, 'type': param_tokens[0]}
            if len(param_tokens) == 1:
                param['name'] = "arg%d" % i
            else:
                if len(param_tokens) == 3:
                    param['indexed'] = True
                param['name'] = param_tokens[-1]
            parameters.append(param)

        event_type = '%s(%s)' % (event_name, ','.join([x['type'] for x in parameters]))
        signature = '0x{}'.format(hexlify(eth_utils.keccak(event_type.encode())).decode())

        return {"event_name": event_name, "signature": signature, "parameters": parameters}

    @staticmethod
    def decode_log(event: dict, log):
        """
        Decode log
        :param event:
        :param log:
        :return:
        """
        event_name = event['event_name']
        parameters = event['parameters']
        signature = event['signature']

        if log['topics'][0].hex() == signature:
            res = {
                "event_name": event_name,
                'blockNumber': log['blockNumber'],
                'transactionHash': log['transactionHash']
                }
            if parameters:
                j = 1
                for param in parameters:
                    if param['indexed']:
                        value = eth_abi.decode_single(param['type'], log['topics'][j])
                        j += 1
                        if isinstance(value, bytes):
                            res[param.name] = '0x%r' % hexlify(value.decode())
                        else:
                            res[param.name] = value

                unindexed_types = [x['type'] for x in parameters if not x['indexed']]
                if unindexed_types:
                    unindexed_values = eth_abi.decode_abi(
                        unindexed_types, unhexlify(log['data'][2:]))
                    unindexed_names = [x['name'] for x in parameters if not x['indexed']]
                    for i in range(len(unindexed_values)):
                        if isinstance(unindexed_values[i], bytes):
                            res[unindexed_names[i]] = '0x{}'.format(
                                hexlify(unindexed_values[i]).decode())
                        else:
                            res[unindexed_names[i]] = unindexed_values[i]
            return res
        return None

    @staticmethod
    def decode_logs(events, logs):
        """
        Decode logs in receipt
        :param events: example ['LogAdded(bytes, bytes)]
        :param logs:
        :return:
        """
        ret = []
        for log in logs:
            for event in events:
                try:
                    rret = SmartContractFuncCall.decode_log(event, log)
                    if rret:
                        ret.append(rret)
                        continue
                except Exception:
                    pass
        return ret


class SmartContract:
    """Contract Base"""
    def __init__(self, w3h: W3Helper, contract_addr: str, queue_size: int):
        self.log = logging.getLogger(self.__class__.__name__)
        self.w3h = w3h
        self.contract_addr = Web3.toChecksumAddress(contract_addr)
        self.attrs = {}  # type: dict
        self.events_queue = queue.Queue(queue_size)  # type: queue.Queue
        self._thread_manager = ThreadManager(
            _queue=self.events_queue,
            max_workers=CONTRACT_MAX_TASK_NUM,
            thread_name_prefix=CONTRACT_TASK_NAME_PREFIX
        )

    def __getattr__(self, name):
        if name in self.attrs:
            attr = self.attrs[name]
            if isinstance(attr, str) and ' ' in attr:
                attr = attr.replace(' ', '', -1)
            return SmartContractFuncCall(self.w3h, self.contract_addr, attr)
        raise AttributeError

    def watch(self, event: str, _from: int):
        """
        watch create a new thread to monitor chain event.
        :param event:
        :param _from: monitor from this block number
        :return:
        """
        self.log.debug("Start to watch event %s", event)
        parsed_event = SmartContractFuncCall.parse_event_signature(event)
        topics = [parsed_event['signature']]
        if _from is None:
            _from = self.w3h.blockNumber + 1
        curr_block = self.w3h.blockNumber
        event_filter = {
            "address": self.contract_addr,
            "topics": topics,
            "fromBlock": hex(_from),
            "toBlock": hex(curr_block)
        }
        while not self._thread_manager.has_shutdown():
            if _from > curr_block:
                time.sleep(CHAIN_EVENT_GET_FAILED_WAIT_TIME_IN_S)
            else:
                logs = self.w3h.eth.getLogs(event_filter)
                events = SmartContractFuncCall.decode_logs([parsed_event], logs)
                if events:
                    for evt in events:
                        self.log.debug("Got event %s", evt)
                        self.events_queue.put(evt)
                else:
                    time.sleep(CHAIN_EVENT_GET_FAILED_WAIT_TIME_IN_S)
                _from = curr_block + 1
            curr_block = self.w3h.blockNumber
            event_filter['fromBlock'] = hex(_from)
            event_filter['toBlock'] = hex(curr_block)
        self.log.debug("Stop to watch event %s", event)

    def exit_all_watch(self, wait=False):
        self._thread_manager.shutdown(wait=wait)


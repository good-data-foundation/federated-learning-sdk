"""
This file tests the api that includes get_query_info() and query_completed()
I create the result by mock, then judge whether they are equal
"""
from concurrent import futures
from unittest import mock
import unittest
import datetime
import grpc
from goodDataML.connection import gooddata
from goodDataML.connection.proto import gooddata_service_pb2, gooddata_service_pb2_grpc


def mock_result_get_query_info():
    time = str(datetime.date.today())
    mock_response = gooddata_service_pb2.GetQueryInfoResponse()
    query_info = mock_response.query_info
    query_info.uuid = 'uuid ' + time
    query_info.content = ('content ' + time).encode('utf8')

    query_cluster = mock_response.query_cluster
    query_cluster.query_uuid = 'query_cluster.query_uuid ' + time

    mpc_nodes = query_cluster.mpc_nodes.add()
    mpc_nodes.address = 'mpc_nodes.address1 ' + time
    mpc_nodes.port = 'mpc_nodes.port1 ' + time
    mpc_nodes = query_cluster.mpc_nodes.add()
    mpc_nodes.address = 'mpc_nodes.address2 ' + time
    mpc_nodes.port = 'mpc_nodes.port2 ' + time

    do_nodes = query_cluster.do_nodes.add()
    do_nodes.address = 'do_nodes.address1 ' + time
    do_nodes.port = 'do_nodes.port1 ' + time
    do_nodes = query_cluster.do_nodes.add()
    do_nodes.address = 'do_nodes.address2 ' + time
    do_nodes.port = 'do_nodes.port2 ' + time

    return mock_response


def mock_result_query_completed():
    mock_response = gooddata_service_pb2.QueryCompletedResponse()
    return mock_response


def mock_result_register_do():
    mock_response = gooddata_service_pb2.RegisterDOResponse()
    mock_response.uuid = 'test_do_uuid'
    return mock_response


def is_right_query_completed(result):
    return result == gooddata_service_pb2.QueryCompletedResponse()


def is_right_get_query_info(result):
    time = str(datetime.date.today())
    return bool(result.query_info.uuid == 'uuid ' + time and
                result.query_info.content == ('content ' + time).encode('utf8') and
                result.query_cluster.query_uuid == 'query_cluster.query_uuid ' + time and
                result.query_cluster.mpc_nodes[0].address == 'mpc_nodes.address1 ' + time and
                result.query_cluster.mpc_nodes[0].port == 'mpc_nodes.port1 ' + time and
                result.query_cluster.mpc_nodes[1].address == 'mpc_nodes.address2 ' + time and
                result.query_cluster.mpc_nodes[1].port == 'mpc_nodes.port2 ' + time and
                result.query_cluster.do_nodes[0].address == 'do_nodes.address1 ' + time and
                result.query_cluster.do_nodes[0].port == 'do_nodes.port1 ' + time and
                result.query_cluster.do_nodes[1].address == 'do_nodes.address2 ' + time and
                result.query_cluster.do_nodes[1].port == 'do_nodes.port2 ' + time)


class FakeServicer(gooddata_service_pb2_grpc.GoodDataServiceServicer):
    def GetQueryInfo(self, request, context):
        requestid = request.query_uuid  # test signal
        tmp = gooddata_service_pb2.GetQueryInfoResponse()

        # create testing response by adding test signal
        query_info = tmp.query_info
        query_info.uuid = 'uuid ' + requestid
        query_info.content = ('content ' + requestid).encode('utf8')

        query_cluster = tmp.query_cluster
        query_cluster.query_uuid = 'query_cluster.query_uuid ' + requestid

        mpc_nodes = query_cluster.mpc_nodes.add()
        mpc_nodes.address = 'mpc_nodes.address1 ' + requestid
        mpc_nodes.port = 'mpc_nodes.port1 ' + requestid
        mpc_nodes = query_cluster.mpc_nodes.add()
        mpc_nodes.address = 'mpc_nodes.address2 ' + requestid
        mpc_nodes.port = 'mpc_nodes.port2 ' + requestid

        do_nodes = query_cluster.do_nodes.add()
        do_nodes.address = 'do_nodes.address1 ' + requestid
        do_nodes.port = 'do_nodes.port1 ' + requestid
        do_nodes = query_cluster.do_nodes.add()
        do_nodes.address = 'do_nodes.address2 ' + requestid
        do_nodes.port = 'do_nodes.port2 ' + requestid

        yield tmp

    def QueryCompleted(self, request, context):
        return gooddata_service_pb2.QueryCompletedResponse()

    def RegisterDO(self, request, context):
        response = gooddata_service_pb2.RegisterDOResponse(uuid='test_uuid')
        return response


class TestReceiveFromSs(unittest.TestCase):
    def setUp(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        gooddata_service_pb2_grpc.add_GoodDataServiceServicer_to_server(FakeServicer(), self.server)
        self.server.add_insecure_port('[::]:9001')
        self.server.start()

    def test_get_query_info(self):
        with grpc.insecure_channel('localhost:9001') as channel:
            response = gooddata.get_query_info('test_query_uuid', channel)
            self.assertIsNotNone(response, 'get_query_info error!')

    def test_query_completed(self):
        query_execution_info_dict = {'_uuid': 'test_uuid',
                                     'start_time_ns': 66,
                                     'finished_time_ns': 666,
                                     'status': 1}
        with grpc.insecure_channel('localhost:9001') as channel:
            response = gooddata.query_completed('test_do_uuid',
                                                'tmp_model/testmodel.pt',
                                                channel,
                                                query_execution_info_dict
                                                )
            self.assertIsNotNone(response, 'query_completed error!')

    def test_register_do(self):
        with grpc.insecure_channel('localhost:9001') as channel:
            node_info_do = {'address': '127.0.0.2', 'port': '1235', '_uuid': 'donode'}
            do_uuid = gooddata.register_do('test_do_uuid', 'test_key2', channel, node_info_do).uuid
            self.assertIsInstance(do_uuid, str)


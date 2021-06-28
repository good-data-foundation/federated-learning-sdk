""" receive data from GoodDataService by grpc"""
import os
import time
import grpc

from goodDataML.connection.utils.ml_utils import bytes_from_file
from goodDataML.connection.proto import gooddata_service_pb2
from goodDataML.connection.proto import gooddata_service_pb2_grpc


# query_uuid:string, channel is created in main scrpit
def get_query_info(query_uuid: str, channel: grpc.insecure_channel):
    # build rpc connection with GoodData Service
    stub = gooddata_service_pb2_grpc.GoodDataServiceStub(channel)

    getqueryinfo_response = stub.GetQueryInfo(gooddata_service_pb2.GetQueryInfoRequest(query_uuid=query_uuid))

    filename_start = "./tmp/model_%s_start_%s.pt" % (query_uuid, time.time())

    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')

    req = None
    for req in getqueryinfo_response:
        with open(filename_start, 'ab') as file:
            file.write(req.query_info.content)
    model_path = os.path.abspath(filename_start)

    return req, model_path


def gen_query_completed(do_uuid: str, content, query_execution_info_dict: dict):
    for chunk in content:
        if not chunk:
            return
        req = gooddata_service_pb2.QueryCompletedRequest()
        req.do_uuid = do_uuid
        req.query_execution_info.uuid = query_execution_info_dict['_uuid']
        req.query_execution_info.start_time_ns = query_execution_info_dict['start_time_ns']
        req.query_execution_info.finished_time_ns = query_execution_info_dict['finished_time_ns']
        req.query_execution_info.status = query_execution_info_dict['status']
        req.query_execution_info.result = chunk
        yield req


def query_completed(do_uuid: str, result_path: str, channel: grpc.insecure_channel, query_execution_info_dict: dict):
    stub = gooddata_service_pb2_grpc.GoodDataServiceStub(channel)

    bytes_content = bytes_from_file(result_path, 3 * 1024 * 1024)
    req = gen_query_completed(do_uuid, bytes_content, query_execution_info_dict)
    query_completed_response = stub.QueryCompleted(req)

    return query_completed_response


def register_do(public_key: str, channel: grpc.insecure_channel, node_info: dict):
    stub = gooddata_service_pb2_grpc.GoodDataServiceStub(channel)
    req = gooddata_service_pb2.RegisterDORequest()
    do_info = req.do_info
    do_info.public_key = public_key
    node = do_info.node_info
    # if the field name is 'uuid', it will get error because of python's key word?
    node.address = node_info['address']
    node.port = node_info['port']

    res = stub.RegisterDO(req)

    return res


def register_qc(uuid: str, public_key: str, channel: grpc.insecure_channel, node_info: dict):
    stub = gooddata_service_pb2_grpc.GoodDataServiceStub(channel)
    req = gooddata_service_pb2.RegisterQCRequest()
    do_info = req.qc_info
    do_info.uuid = uuid
    do_info.public_key = public_key
    node = do_info.node_info
    node.uuid = node_info['_uuid']
    node.address = node_info['address']
    node.port = node_info['port']

    res = stub.RegisterQC(req)

    return res


def get_query_execution_info(query_uuid: str, channel: grpc.insecure_channel):
    stub = gooddata_service_pb2_grpc.GoodDataServiceStub(channel)
    req = gooddata_service_pb2.GetQueryExecutionInfoRequest(query_uuid=query_uuid)

    responses = stub.GetQueryExecutionInfo(req)
    filename = './tmp/finished_%s_model.pt' % query_uuid

    res = None
    for res in responses:
        with open(filename, 'ab') as file:
            file.write(res.query_execution_info.result)
    model_path = os.path.abspath(filename)

    return res, model_path


def test_model_completed(test_uuid: str, channel: grpc.insecure_channel):
    stub = gooddata_service_pb2_grpc.GoodDataServiceStub(channel)
    req = gooddata_service_pb2.TestModelCompletedRequest(test_uuid=test_uuid)

    response = stub.TestModelCompleted(req)
    return response


def register_mpc(mpc_info: dict, channel: grpc.insecure_channel):
    stub = gooddata_service_pb2_grpc.GoodDataServiceStub(channel)
    public_key = mpc_info['public_key']
    _node_info = mpc_info['node_info']

    resquest = gooddata_service_pb2.RegisterMPCRequest()
    mpc_info = resquest.mpc_info
    mpc_info.public_key = public_key
    node_info = mpc_info.node_info
    node_info.address = _node_info['address']
    node_info.port = _node_info['port']

    res = stub.RegisterMPC(resquest)

    return res

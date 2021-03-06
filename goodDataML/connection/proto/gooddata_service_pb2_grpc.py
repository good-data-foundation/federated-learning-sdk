# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from goodDataML.connection.proto import gooddata_service_pb2 as gooddata__service__pb2


class GoodDataServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RegisterQC = channel.unary_unary(
                '/proto.GoodDataService/RegisterQC',
                request_serializer=gooddata__service__pb2.RegisterQCRequest.SerializeToString,
                response_deserializer=gooddata__service__pb2.RegisterQCResponse.FromString,
                )
        self.LoginQC = channel.unary_unary(
                '/proto.GoodDataService/LoginQC',
                request_serializer=gooddata__service__pb2.LoginQCRequest.SerializeToString,
                response_deserializer=gooddata__service__pb2.LoginQCResponse.FromString,
                )
        self.RegisterDO = channel.unary_unary(
                '/proto.GoodDataService/RegisterDO',
                request_serializer=gooddata__service__pb2.RegisterDORequest.SerializeToString,
                response_deserializer=gooddata__service__pb2.RegisterDOResponse.FromString,
                )
        self.RegisterMPC = channel.unary_unary(
                '/proto.GoodDataService/RegisterMPC',
                request_serializer=gooddata__service__pb2.RegisterMPCRequest.SerializeToString,
                response_deserializer=gooddata__service__pb2.RegisterMPCResponse.FromString,
                )
        self.SubmitQuery = channel.stream_unary(
                '/proto.GoodDataService/SubmitQuery',
                request_serializer=gooddata__service__pb2.SubmitQueryRequest.SerializeToString,
                response_deserializer=gooddata__service__pb2.SubmitQueryResponse.FromString,
                )
        self.GetQueryInfo = channel.unary_stream(
                '/proto.GoodDataService/GetQueryInfo',
                request_serializer=gooddata__service__pb2.GetQueryInfoRequest.SerializeToString,
                response_deserializer=gooddata__service__pb2.GetQueryInfoResponse.FromString,
                )
        self.QueryCompleted = channel.stream_unary(
                '/proto.GoodDataService/QueryCompleted',
                request_serializer=gooddata__service__pb2.QueryCompletedRequest.SerializeToString,
                response_deserializer=gooddata__service__pb2.QueryCompletedResponse.FromString,
                )
        self.GetQueryExecutionInfo = channel.unary_stream(
                '/proto.GoodDataService/GetQueryExecutionInfo',
                request_serializer=gooddata__service__pb2.GetQueryExecutionInfoRequest.SerializeToString,
                response_deserializer=gooddata__service__pb2.GetQueryExecutionInfoResponse.FromString,
                )
        self.TestModel = channel.unary_unary(
                '/proto.GoodDataService/TestModel',
                request_serializer=gooddata__service__pb2.TestModelRequest.SerializeToString,
                response_deserializer=gooddata__service__pb2.TestModelResponse.FromString,
                )
        self.TestModelCompleted = channel.unary_unary(
                '/proto.GoodDataService/TestModelCompleted',
                request_serializer=gooddata__service__pb2.TestModelCompletedRequest.SerializeToString,
                response_deserializer=gooddata__service__pb2.TestModelCompletedResponse.FromString,
                )


class GoodDataServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RegisterQC(self, request, context):
        """RegisterQC Register query customer, generate UUID, notify chain and return UUID.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def LoginQC(self, request, context):
        """LoginQC login query customer
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterDO(self, request, context):
        """RegisterDO Register data owner, generate UUID, notify chain and return UUID.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterMPC(self, request, context):
        """RegisterMPC Register MPC node, generate UUID, notify chain and return UUID.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SubmitQuery(self, request_iterator, context):
        """SubmitQuery Query customer submits the query. GDS sevice will save query info,
        chose DO and MPC nodes and notify the chain.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetQueryInfo(self, request, context):
        """GetQueryInfo DO gets query information after it is chosen for query training. 
        Returns QueryInfo as well as chosen nodes info for P2P connection.
        Returns NOT_FOUND error in cases query is not found.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def QueryCompleted(self, request_iterator, context):
        """QueryCompleted DO sends query result after it completes the query training. 
        GDS service will check query results and save query info in case all of DO
        results are received and consistent.
        Returns UNKNOWN error in case query results are not consistent.
        Returns NOT_FOUND error in case query is not found.   
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetQueryExecutionInfo(self, request, context):
        """GetQueryExecutionInfo Query customer asks for query result.
        Returns NOT_FOUND error in case query is not found.  
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def TestModel(self, request, context):
        """TestModel initiate a test.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def TestModelCompleted(self, request, context):
        """TestModelCompleted completed a test.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GoodDataServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RegisterQC': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterQC,
                    request_deserializer=gooddata__service__pb2.RegisterQCRequest.FromString,
                    response_serializer=gooddata__service__pb2.RegisterQCResponse.SerializeToString,
            ),
            'LoginQC': grpc.unary_unary_rpc_method_handler(
                    servicer.LoginQC,
                    request_deserializer=gooddata__service__pb2.LoginQCRequest.FromString,
                    response_serializer=gooddata__service__pb2.LoginQCResponse.SerializeToString,
            ),
            'RegisterDO': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterDO,
                    request_deserializer=gooddata__service__pb2.RegisterDORequest.FromString,
                    response_serializer=gooddata__service__pb2.RegisterDOResponse.SerializeToString,
            ),
            'RegisterMPC': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterMPC,
                    request_deserializer=gooddata__service__pb2.RegisterMPCRequest.FromString,
                    response_serializer=gooddata__service__pb2.RegisterMPCResponse.SerializeToString,
            ),
            'SubmitQuery': grpc.stream_unary_rpc_method_handler(
                    servicer.SubmitQuery,
                    request_deserializer=gooddata__service__pb2.SubmitQueryRequest.FromString,
                    response_serializer=gooddata__service__pb2.SubmitQueryResponse.SerializeToString,
            ),
            'GetQueryInfo': grpc.unary_stream_rpc_method_handler(
                    servicer.GetQueryInfo,
                    request_deserializer=gooddata__service__pb2.GetQueryInfoRequest.FromString,
                    response_serializer=gooddata__service__pb2.GetQueryInfoResponse.SerializeToString,
            ),
            'QueryCompleted': grpc.stream_unary_rpc_method_handler(
                    servicer.QueryCompleted,
                    request_deserializer=gooddata__service__pb2.QueryCompletedRequest.FromString,
                    response_serializer=gooddata__service__pb2.QueryCompletedResponse.SerializeToString,
            ),
            'GetQueryExecutionInfo': grpc.unary_stream_rpc_method_handler(
                    servicer.GetQueryExecutionInfo,
                    request_deserializer=gooddata__service__pb2.GetQueryExecutionInfoRequest.FromString,
                    response_serializer=gooddata__service__pb2.GetQueryExecutionInfoResponse.SerializeToString,
            ),
            'TestModel': grpc.unary_unary_rpc_method_handler(
                    servicer.TestModel,
                    request_deserializer=gooddata__service__pb2.TestModelRequest.FromString,
                    response_serializer=gooddata__service__pb2.TestModelResponse.SerializeToString,
            ),
            'TestModelCompleted': grpc.unary_unary_rpc_method_handler(
                    servicer.TestModelCompleted,
                    request_deserializer=gooddata__service__pb2.TestModelCompletedRequest.FromString,
                    response_serializer=gooddata__service__pb2.TestModelCompletedResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'proto.GoodDataService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class GoodDataService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RegisterQC(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/proto.GoodDataService/RegisterQC',
            gooddata__service__pb2.RegisterQCRequest.SerializeToString,
            gooddata__service__pb2.RegisterQCResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def LoginQC(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/proto.GoodDataService/LoginQC',
            gooddata__service__pb2.LoginQCRequest.SerializeToString,
            gooddata__service__pb2.LoginQCResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RegisterDO(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/proto.GoodDataService/RegisterDO',
            gooddata__service__pb2.RegisterDORequest.SerializeToString,
            gooddata__service__pb2.RegisterDOResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RegisterMPC(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/proto.GoodDataService/RegisterMPC',
            gooddata__service__pb2.RegisterMPCRequest.SerializeToString,
            gooddata__service__pb2.RegisterMPCResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SubmitQuery(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/proto.GoodDataService/SubmitQuery',
            gooddata__service__pb2.SubmitQueryRequest.SerializeToString,
            gooddata__service__pb2.SubmitQueryResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetQueryInfo(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/proto.GoodDataService/GetQueryInfo',
            gooddata__service__pb2.GetQueryInfoRequest.SerializeToString,
            gooddata__service__pb2.GetQueryInfoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def QueryCompleted(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/proto.GoodDataService/QueryCompleted',
            gooddata__service__pb2.QueryCompletedRequest.SerializeToString,
            gooddata__service__pb2.QueryCompletedResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetQueryExecutionInfo(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/proto.GoodDataService/GetQueryExecutionInfo',
            gooddata__service__pb2.GetQueryExecutionInfoRequest.SerializeToString,
            gooddata__service__pb2.GetQueryExecutionInfoResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def TestModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/proto.GoodDataService/TestModel',
            gooddata__service__pb2.TestModelRequest.SerializeToString,
            gooddata__service__pb2.TestModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def TestModelCompleted(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/proto.GoodDataService/TestModelCompleted',
            gooddata__service__pb2.TestModelCompletedRequest.SerializeToString,
            gooddata__service__pb2.TestModelCompletedResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

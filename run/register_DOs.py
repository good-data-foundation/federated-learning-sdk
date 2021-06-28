import grpc

from goodDataML.connection.gooddata import register_do, register_mpc
from goodDataML.learning.testnet.utils import get_config_file

if __name__ == '__main__':
    config = get_config_file('./local_config.json')

    good_data_service_channel = grpc.insecure_channel(config['good_data_service_address'],
                                                      options=[
                                                          ('grpc.max_send_message_length', 5 * 1024 * 1024),
                                                          ('grpc.max_receive_message_length', 5 * 1024 * 1024),
                                                      ])

    for do_name in config["dos"]:
        do_config = config[do_name]
        register_do(public_key=do_config['public_key'],
                    channel=good_data_service_channel,
                    node_info={'address': do_config['ip_address'], 'port': do_config['port']})

    for mpc_name in config["mpcs"]:
        mpc_config = config[mpc_name]
        register_mpc(channel=good_data_service_channel,
                     mpc_info={'public_key': mpc_config['public_key'],
                               'node_info': {'address': mpc_config['ip_address'], 'port': mpc_config['port']}})

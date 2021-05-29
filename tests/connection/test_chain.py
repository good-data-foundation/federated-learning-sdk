import random
import threading
import time
import unittest
import logging
from goodDataML.connection.chain import ChainEvent, GoodDataChain
from goodDataML.connection.chain_connection_helpers import CHAIN_EVENT_GET_FAILED_WAIT_TIME_IN_S


class TestChain(unittest.TestCase):
    def test_chain_event_call_back(self):
        expect_count = 3
        expect_log_added_list = []
        expect_query_submitted_list = []
        expect_prediction_submitted_list = []
        actual_log_added_list = []
        actual_query_submitted_list = []
        actual_prediction_submitted_list = []

        ctx = {
            'config1': {'ip': '127.0.0.1', 'port': '80'},
            'config2': {'ip': '127.0.0.1', 'port': '8080'},
        }

        # implement your callback
        def consumer_log_added(chain, event):
            self.assertEqual(chain.context, ctx)
            actual_log_added_list.append(event)

        def consumer_query_submitted(chain, event):
            self.assertEqual(chain.context, ctx)
            actual_query_submitted_list.append(event)

        def consumer_prediction_submitted(chain, event):
            self.assertEqual(chain.context, ctx)
            actual_prediction_submitted_list.append(event)

        sc = GoodDataChain('http://192.168.1.5:8545',
                           'ed587757a0ce2b5a6b21b8b5e72be2646ad80122a638539c9420b6b9cb9e0638',
                           ctx)
        sc.subscribe(ChainEvent.LogAdded, consumer_log_added)

        sc.subscribe(ChainEvent.LogAdded, consumer_log_added)
        sc.subscribe(ChainEvent.QuerySubmitted, consumer_query_submitted)
        sc.subscribe(ChainEvent.PredictionSubmitted, consumer_prediction_submitted)

        # ensure GoodDataChain is watching all event

        # produce event
        def producer():
            def get_events(_event_type, receipt):
                return receipt['events']

            count = expect_count
            while count > 0:
                print('producer...')
                rand = random.randint(1, 10000)
                data = f'{rand}'

                receipt = sc.add_log(data, {"data": data})
                print('receipt:', receipt)
                log_added_events = get_events(ChainEvent.LogAdded, receipt)
                expect_log_added_list.extend(log_added_events)

                receipt = sc.submit_query(data.encode(), data.encode(), rand//5000, rand)
                query_submitted_events = get_events(ChainEvent.QuerySubmitted, receipt)
                expect_query_submitted_list.extend(query_submitted_events)

                receipt = sc.submit_prediction(data.encode(), data.encode())
                prediction_submitted_events = get_events(ChainEvent.PredictionSubmitted, receipt)
                expect_prediction_submitted_list.extend(prediction_submitted_events)

                time.sleep(1)
                count -= 1

        producer()
        # Don't forget time sleep for unit test.
        # It ensure all events will be watched.
        time.sleep(CHAIN_EVENT_GET_FAILED_WAIT_TIME_IN_S + 1)

        def sort_by_block_number(d: dict):
            if 'blockNumber' in d:
                return d['blockNumber']
            else:
                raise KeyError

        actual_log_added_list.sort(key=sort_by_block_number)
        actual_query_submitted_list.sort(key=sort_by_block_number)
        actual_prediction_submitted_list.sort(key=sort_by_block_number)

        expect_log_added_list.sort(key=sort_by_block_number)
        expect_query_submitted_list.sort(key=sort_by_block_number)
        expect_prediction_submitted_list.sort(key=sort_by_block_number)

        self.assertNotEqual(len(expect_log_added_list), 0)
        self.assertEqual(expect_log_added_list, actual_log_added_list)

        self.assertNotEqual(len(expect_query_submitted_list), 0)
        self.assertEqual(expect_query_submitted_list, actual_query_submitted_list)

        self.assertNotEqual(len(expect_prediction_submitted_list), 0)
        self.assertEqual(expect_prediction_submitted_list, actual_prediction_submitted_list)

        print('before program exit, thread count %d' % threading.active_count())
        sc.shutdown(wait=True)
        self.assertEqual(threading.active_count(), 1)
        print('program successful exit, thread count %d' % threading.active_count())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()

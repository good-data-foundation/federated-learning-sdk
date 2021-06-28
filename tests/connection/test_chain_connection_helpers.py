import unittest
from goodDataML.connection.chain_connection_helpers import *
import threading
import queue


class TestChainConnectionHelpers(unittest.TestCase):
    def test_chain_task_manager_stop_common_thread(self):
        expect_list = [1, 2, 3, 4]
        actual_list = []
        actual_queue = queue.Queue()

        def task(i):
            print('put:', i)
            actual_queue.put(i)
        mgr = ThreadManager()
        for i in range(1, 5):
            mgr.submit(task, i)

        time.sleep(2)
        mgr.shutdown(wait=True)
        while not actual_queue.empty():
            actual_list.append(actual_queue.get())
        actual_list.sort()

        print('expect_list:', expect_list)
        print('actual_list:', actual_list)
        self.assertEqual(expect_list, actual_list)
        self.assertEqual(threading.active_count(), 1)

    def test_chain_task_manager_stop_while_true_thread(self):
        quited_tasks = queue.Queue()
        expect_list = [1, 2, 3, 4]
        actual_list = []
        mgr = ThreadManager()

        def task(i):
            while mgr.has_shutdown():
                time.sleep(1)
            quited_tasks.put(i)

        for i in range(1, 5):
            mgr.submit(task, i)

        time.sleep(2)
        mgr.shutdown(True)

        while not quited_tasks.empty():
            actual_list.append(quited_tasks.get())
        actual_list.sort()
        print('expect_list:', expect_list)
        print('actual_list:', actual_list)
        self.assertEqual(expect_list, actual_list)
        self.assertEqual(threading.active_count(), 1)


if __name__ == '__main__':
    unittest.main()

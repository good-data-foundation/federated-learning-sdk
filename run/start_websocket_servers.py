import subprocess
import sys
import time
from pathlib import Path

from goodDataML.learning.testnet.utils import get_config_file

python = Path(sys.executable).name

FILE_PATH = Path(__file__).resolve().parents[4].joinpath(r"./run_websocket_servers.py")

config = get_config_file('./local_config.json')

call_worker1 = [python, FILE_PATH.name, "--port", config['do1']['port'], "--id", "worker1"]

call_worker2 = [python, FILE_PATH.name, "--port", config['do2']['port'], "--id", "worker2"]

call_mpc1 = [python, FILE_PATH.name, "--port", config['mpc1']['port'], "--id", "mpc1"]

call_mpc2 = [python, FILE_PATH.name, "--port", config['mpc2']['port'], "--id", "mpc2"]

call_mpc3 = [python, FILE_PATH.name, "--port", config['mpc3']['port'], "--id", "mpc3"]


subprocess.Popen(call_worker1)
subprocess.Popen(call_worker2)


subprocess.Popen(call_mpc1)
subprocess.Popen(call_mpc2)
subprocess.Popen(call_mpc3)

while True:
    time.sleep(1)

import subprocess
import sys
import time
from pathlib import Path

python = Path(sys.executable).name

FILE_PATH = Path(__file__).resolve().parents[4].joinpath(r"./run_websocket_servers.py")
print("file_path: ", FILE_PATH)


call_worker1 = [python, FILE_PATH.name, "--port", "10080", "--id", "worker1"]

call_worker2 = [python, FILE_PATH.name, "--port", "10081", "--id", "worker2"]

call_mpc1 = [python, FILE_PATH.name, "--port", "10082", "--id", "mpc1"]

call_mpc2 = [python, FILE_PATH.name, "--port", "10083", "--id", "mpc2"]

call_mpc3 = [python, FILE_PATH.name, "--port", "10084", "--id", "mpc3"]


subprocess.Popen(call_worker1)
subprocess.Popen(call_worker2)


subprocess.Popen(call_mpc1)
subprocess.Popen(call_mpc2)
subprocess.Popen(call_mpc3)

while True:
    time.sleep(1)

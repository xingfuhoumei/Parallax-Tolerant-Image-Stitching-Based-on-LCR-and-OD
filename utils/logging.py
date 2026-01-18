import sys
from pathlib import Path

from config import DEFAULT_LOG_FILE, ensure_dir
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


log_path = Path(DEFAULT_LOG_FILE)
ensure_dir(log_path.parent)
sys.stdout = Logger(str(log_path))

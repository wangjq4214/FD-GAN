import os
import sys

from .osutils import mkdir_if_missing


class Logger:
    """
    日志记录, 同时输出到控制台和文件中
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

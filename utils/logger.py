import logging
import os
from utils.envs import logger_path
from utils.common import get_datetime


class Logger:
    def __init__(self):
        self.name = None
        self.datetime = None
        self.level = None
        self.logger =None
        self.ch = None
        self.fh = None
        self.is_setup = False

    def setup_logger(self, name, level=logging.INFO):
        self.name = name
        self.datetime = get_datetime()
        self.level = level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.ch = self.get_stream_handler()
        self.fh = self.get_file_handler(os.path.join(logger_path, '{}_{}.log'.format(self.name, self.datetime)))
        self.logger.addHandler(self.ch)
        self.logger.addHandler(self.fh)
        self.is_setup = True

    def get_file_handler(self, path):
        fh = logging.FileHandler(path)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setLevel(self.level)
        fh.setFormatter(formatter)
        return fh

    def get_stream_handler(self):
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        ch.setLevel(self.level)
        ch.setFormatter(formatter)
        return ch

    def info(self, msg):
        assert self.is_setup, "Please Setup the Logger First"
        return self.logger.info(msg)

    def debug(self, msg):
        assert self.is_setup, "Please Setup the Logger First"
        return self.logger.debug(msg)

    def error(self, msg):
        assert self.is_setup, "Please Setup the Logger First"
        return self.logger.error(msg)

    def warning(self, msg):
        assert self.is_setup, "Please Setup the Logger First"
        return self.logger.warning(msg)


logger = Logger()
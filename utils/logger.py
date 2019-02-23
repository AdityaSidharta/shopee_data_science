import logging

import torch.nn.functional as F

from utils.envs import logger_repo


def setup_logger(name: str, log_path: str) -> logging.Logger:
    """Logger utility for logging messages

    Prints logs to screen via ch (channel handler), and saves logs to via fh (file handler)

    Args:
        name: Name of logger
        log_path: Path to write logs to

    Returns:
        Logger with logging to both screen output and log file in log path

    Examples:
        >>> type(setup_logger('test_logger', 'test.log'))
        <class 'logging.Logger'>
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    # create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    # create file handler that logs debug messages
    fh = logging.FileHandler("{}".format(log_path))
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)

    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


logger = setup_logger("__name__", logger_repo)


def debug_pred_target(prediction, target):
    logger.debug("target : {}".format(target[:10, :]))
    logger.debug("prediction : {}".format(F.sigmoid(prediction)[:10, :]))
    logger.debug("==========================================")

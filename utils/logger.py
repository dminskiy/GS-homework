import sys
import logging

LOG_FORMAT = "%(asctime)s %(name)s [%(levelname)s] %(message)s"


def setup_logger(name):
    # Create a custom logger
    logger = logging.getLogger(name)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)

    c_format = logging.Formatter(LOG_FORMAT)
    c_handler.setFormatter(c_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.setLevel(logging.DEBUG)

    return logger

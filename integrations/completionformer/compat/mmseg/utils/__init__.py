import logging


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger("completionformer")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger

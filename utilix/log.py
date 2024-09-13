import logging


HANDLER = None


def setup_logger(logging_level="WARNING"):
    global HANDLER
    logger = logging.getLogger("utilix")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if HANDLER is not None:
        logger.removeHandler(HANDLER)
    HANDLER = logging.StreamHandler()
    HANDLER.setLevel(logging_level)
    HANDLER.setFormatter(formatter)
    logger.setLevel(logging_level)
    logger.addHandler(HANDLER)
    return logger

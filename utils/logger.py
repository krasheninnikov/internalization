import logging
from transformers import logging as hf_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# hf_logging.set_verbosity_info()
hf_logging.set_verbosity_warning() # this disables GenerationConfig messages


def setup_logger(name):
    logger = logging.getLogger(name)
    return logger

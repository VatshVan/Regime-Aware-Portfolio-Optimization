import gc
import logging
from pathlib import Path

def clear_memory():
    """Explicitly invoke garbage collection to clear unreferenced memory."""
    gc.collect()

def setup_logging(name: str) -> logging.Logger:
    """Configures and returns a standard logger for Stage 3 modules."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        # Create console handler with standard format
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

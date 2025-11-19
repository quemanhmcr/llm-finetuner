# -*- coding: utf-8 -*-
"""
Main entry point for the finetune module.

This module allows the package to be executed with:
python -m finetune.src
"""

import logging
from .cli import app
from .utils import setup_logging

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Advanced Professional ML Fine-tuning System")
    app()

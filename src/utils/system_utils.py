# -*- coding: utf-8 -*-
"""
System utilities for ML fine-tuning system.

This module provides system-related utilities including GPU checking
and reproducibility settings.
"""

import logging
import random
import numpy as np
import torch
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """
    Set seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"✓ Seed set to {seed}")


def check_gpu_availability() -> bool:
    """
    Check and log GPU information.
    
    Returns:
        True if GPU is available, False otherwise
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"✓ Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        return True
    else:
        logger.warning("⚠ No GPU found - training will be very slow!")
        return False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Optional device specification ("cuda", "cpu", or "auto")
        
    Returns:
        PyTorch device object
    """
    if device == "cpu":
        return torch.device("cpu")
    elif device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device == "auto" or device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        logger.warning(f"Device {device} not available, falling back to CPU")
        return torch.device("cpu")

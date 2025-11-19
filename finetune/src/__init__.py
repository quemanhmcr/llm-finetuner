# -*- coding: utf-8 -*-
"""
Advanced Professional ML Fine-tuning System - 2025 Optimized Edition

====================================================================

A production-grade, research-backed implementation for fine-tuning small LLMs with:

- State-of-the-art PEFT: LoRA, QLoRA, DoRA with optimal hyperparameters
- Advanced RL: DPO, PPO, GRPO (Group Relative Policy Optimization)
- Reasoning-focused datasets: GSM8K, MetaMathQA, NuminaMath
- Performance optimizations: Flash Attention 2, gradient checkpointing, mixed precision
- Comprehensive evaluation and monitoring

Based on latest research from:

- DeepSeek-R1 (2025): GRPO and reasoning alignment
- QLoRA paper: 4-bit quantization best practices
- DoRA: Improved weight decomposition
- DPO: Reward-free preference optimization

Example Usage:

-------------

# 1. SFT with optimal LoRA on reasoning data
python -m finetune.src sft --peft-method lora --use-reasoning-data

# 2. QLoRA with 4-bit quantization (memory efficient)
python -m finetune.src sft --peft-method qlora --batch-size 2

# 3. DPO for preference alignment
python -m finetune.src dpo --beta 0.1

# 4. GRPO (DeepSeek-style) - most advanced
python -m finetune.src grpo --learning-rate 5e-6

# 5. Full pipeline: SFT -> DPO
python -m finetune.src pipeline --stages sft,dpo

"""

from .cli import app
from .configs import (
    BaseConfig,
    DataConfig,
    PEFTConfig,
    TrainingConfig,
    DPOTrainingConfig,
    GRPOConfig,
)
from .data import ReasoningDatasetHandler
from .models import load_model_and_tokenizer
from .trainers import SFTTrainerWrapper, DPOTrainerWrapper, GRPOTrainerWrapper
from .utils import setup_logging, get_logger, set_seed, check_gpu_availability, get_device

__version__ = "1.0.0"
__all__ = [
    "app",
    "BaseConfig",
    "DataConfig",
    "PEFTConfig",
    "TrainingConfig",
    "DPOTrainingConfig",
    "GRPOConfig",
    "ReasoningDatasetHandler",
    "load_model_and_tokenizer",
    "SFTTrainerWrapper",
    "DPOTrainerWrapper",
    "GRPOTrainerWrapper",
    "setup_logging",
    "get_logger",
    "set_seed",
    "check_gpu_availability",
    "get_device",
]

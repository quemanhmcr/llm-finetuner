# -*- coding: utf-8 -*-
"""Bộ xử lý dataset suy luận cho hệ thống fine-tuning ML.

Module này cung cấp các lớp để xử lý các dataset tập trung vào suy luận
bao gồm GSM8K, MetaMathQA, và các dataset suy luận toán học khác.
"""

import logging
from typing import Dict, List

from datasets import load_dataset, concatenate_datasets

from finetune.src.configs import DataConfig

logger = logging.getLogger(__name__)


class ReasoningDatasetHandler:
    """
    Handles loading and formatting reasoning datasets.
    Focuses on: GSM8K, MetaMathQA, NuminaMath for mathematical reasoning.
    """
    
    def __init__(self, config: DataConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
    def _format_reasoning_prompt(self, example: Dict) -> str:
        """
        Format for reasoning tasks - encourages step-by-step thinking.
        Based on Chain-of-Thought prompting.
        """
        if "question" in example:  # GSM8K format
            question = example["question"]
            answer = example.get("answer", "")
            
            return f"""You are a helpful AI assistant that solves mathematical problems step by step.

<|end|>

{question}

<|end|>

{answer}

<|end|>"""
        else:
            # Generic format
            instruction = example.get("instruction", "")
            response = example.get("response", "")
            return f"""{instruction}

<|end|>

{response}

<|end|>"""
    
    def get_reasoning_dataset(self):
        """Load and prepare reasoning datasets."""
        logger.info("Loading reasoning datasets...")
        
        datasets = []
        
        # GSM8K - Grade school math
        logger.info("Loading GSM8K (grade school math)...")
        gsm8k = load_dataset("openai/gsm8k", "main", split="train")
        datasets.append(gsm8k)
        logger.info(f"  Loaded {len(gsm8k)} GSM8K examples")
        
        # MetaMathQA - Augmented math reasoning (if available)
        try:
            logger.info("Loading MetaMathQA...")
            metamath = load_dataset("meta-math/MetaMathQA", split="train")
            # Sample subset for efficiency
            if len(metamath) > 50000:
                metamath = metamath.shuffle(seed=42).select(range(50000))
            datasets.append(metamath)
            logger.info(f"  Loaded {len(metamath)} MetaMathQA examples")
        except Exception as e:
            logger.warning(f"Could not load MetaMathQA: {e}")
        
        # Combine datasets
        if len(datasets) > 1:
            combined = concatenate_datasets(datasets)
        else:
            combined = datasets[0]
        
        # Shuffle and limit size
        combined = combined.shuffle(seed=42)
        if self.config.train_size > 0 and self.config.train_size < len(combined):
            combined = combined.select(range(self.config.train_size))
        
        logger.info(f"✓ Total reasoning dataset size: {len(combined)}")
        return combined
    
    def get_dpo_dataset(self):
        """Load DPO preference dataset."""
        logger.info(f"Loading DPO dataset: {self.config.dpo_dataset}")
        
        dataset = load_dataset(
            self.config.dpo_dataset,
            split="train_prefs"
        )
        
        # Sample for efficiency
        if len(dataset) > 10000:
            dataset = dataset.shuffle(seed=42).select(range(10000))
        
        def format_dpo(example):
            """Format for DPO trainer."""
            return {
                "prompt": example["prompt"],
                "chosen": example["chosen"][1]["content"] if isinstance(example["chosen"], list) else example["chosen"],
                "rejected": example["rejected"][1]["content"] if isinstance(example["rejected"], list) else example["rejected"],
            }
        
        dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)
        logger.info(f"✓ Loaded {len(dataset)} DPO preference pairs")
        return dataset

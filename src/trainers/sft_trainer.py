# -*- coding: utf-8 -*-
"""Trainer SFT cho hệ thống fine-tuning ML.

Module này cung cấp lớp trainer Huấn luyện Giám sát
để huấn luyện mô hình trên các dataset suy luận.
"""

import logging
import os
from typing import Any

from transformers import TrainingArguments
from trl import SFTTrainer

from finetune.src.configs import BaseConfig, DataConfig, TrainingConfig
from finetune.src.data import ReasoningDatasetHandler

logger = logging.getLogger(__name__)


class SFTTrainerWrapper:
    """Supervised Fine-Tuning with reasoning datasets."""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        base_config: BaseConfig,
        data_config: DataConfig,
        training_config: TrainingConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.base_config = base_config
        self.data_config = data_config
        self.training_config = training_config
        
        self.output_dir = os.path.join(base_config.output_dir, "sft")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train(self) -> str:
        """
        Run SFT training.
        
        Returns:
            Path to the saved model
        """
        logger.info("=" * 80)
        logger.info("STARTING SUPERVISED FINE-TUNING (SFT)")
        logger.info("=" * 80)
        
        # Load dataset
        dataset_handler = ReasoningDatasetHandler(self.data_config, self.tokenizer)
        train_dataset = dataset_handler.get_reasoning_dataset()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            bf16=self.training_config.bf16,
            fp16=self.training_config.fp16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            optim=self.training_config.optim,
            max_grad_norm=self.training_config.max_grad_norm,
            report_to="none",  # Disable wandb for simplicity
            logging_first_step=True,
            remove_unused_columns=False,
        )
        
        # SFT Trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            max_seq_length=self.data_config.max_seq_length,
            formatting_func=dataset_handler._format_reasoning_prompt,
            args=training_args,
            packing=False,  # Don't pack sequences
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        final_path = os.path.join(self.output_dir, "final_model")
        logger.info(f"Saving final model to {final_path}")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        logger.info("✓ SFT training complete!")
        return final_path

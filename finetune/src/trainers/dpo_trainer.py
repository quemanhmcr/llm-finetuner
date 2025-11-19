# -*- coding: utf-8 -*-
"""Trainer DPO cho hệ thống fine-tuning ML.

Module này cung cấp lớp trainer Tối ưu hóa Ưu tiên Trực tiếp
để căn chỉnh mô hình không cần reward.
"""

import logging
import os
from typing import Any

from trl import DPOTrainer, DPOConfig

from finetune.src.configs import BaseConfig, DataConfig, DPOTrainingConfig
from finetune.src.data import ReasoningDatasetHandler

logger = logging.getLogger(__name__)


class DPOTrainerWrapper:
    """Direct Preference Optimization - Reward-free alignment."""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        base_config: BaseConfig,
        data_config: DataConfig,
        dpo_config: DPOTrainingConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.base_config = base_config
        self.data_config = data_config
        self.dpo_config = dpo_config
        
        self.output_dir = os.path.join(base_config.output_dir, "dpo")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def train(self) -> str:
        """
        Run DPO training.
        
        Returns:
            Path to the saved model
        """
        logger.info("=" * 80)
        logger.info("STARTING DIRECT PREFERENCE OPTIMIZATION (DPO)")
        logger.info("=" * 80)
        
        # Load dataset
        dataset_handler = ReasoningDatasetHandler(self.data_config, self.tokenizer)
        train_dataset = dataset_handler.get_dpo_dataset()
        
        # DPO Training arguments
        training_args = DPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.dpo_config.num_train_epochs,
            per_device_train_batch_size=self.dpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.dpo_config.gradient_accumulation_steps,
            learning_rate=self.dpo_config.learning_rate,
            beta=self.dpo_config.beta,
            logging_steps=10,
            save_steps=500,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            max_length=self.data_config.max_seq_length,
            max_prompt_length=self.data_config.max_seq_length // 2,
        )
        
        # DPO Trainer
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # TRL creates reference model automatically
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )
        
        logger.info("Starting DPO training...")
        trainer.train()
        
        # Save final model
        final_path = os.path.join(self.output_dir, "final_model")
        logger.info(f"Saving final model to {final_path}")
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        logger.info("✓ DPO training complete!")
        return final_path

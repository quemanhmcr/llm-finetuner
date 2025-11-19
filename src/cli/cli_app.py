# -*- coding: utf-8 -*-
"""Ứng dụng CLI cho hệ thống fine-tuning ML.

Module này cung cấp giao diện dòng lệnh để chạy các
phương pháp huấn luyện khác nhau bao gồm SFT, DPO, và GRPO.
"""

import logging
import typer

from finetune.src.configs import (
    BaseConfig,
    DataConfig,
    PEFTConfig,
    TrainingConfig,
    DPOTrainingConfig,
    GRPOConfig,
)
from finetune.src.models import load_model_and_tokenizer
from finetune.src.trainers import SFTTrainerWrapper, DPOTrainerWrapper, GRPOTrainerWrapper
from finetune.src.utils import check_gpu_availability, set_seed

logger = logging.getLogger(__name__)

# Create CLI app
app = typer.Typer(help="Advanced Professional ML Fine-tuning System")


@app.command()
def sft(
    model_name: str = typer.Option("microsoft/Phi-3.5-mini-instruct", help="Base model"),
    peft_method: str = typer.Option("lora", help="PEFT method: lora, qlora, dora"),
    output_dir: str = typer.Option("./results_sft", help="Output directory"),
    epochs: int = typer.Option(3, help="Training epochs"),
    batch_size: int = typer.Option(1, help="Batch size per device"),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
    use_reasoning_data: bool = typer.Option(True, help="Use reasoning datasets"),
    lora_r: int = typer.Option(32, help="LoRA rank"),
):
    """Run Supervised Fine-Tuning with optimal PEFT configuration."""
    
    # Setup
    check_gpu_availability()
    base_config = BaseConfig(model_name=model_name, output_dir=output_dir)
    set_seed(base_config.seed)
    
    # Configs
    data_config = DataConfig(use_reasoning_data=use_reasoning_data)
    peft_config = PEFTConfig(
        peft_method=peft_method,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,  # Keep 2x relationship
    )
    training_config = TrainingConfig(
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(base_config, peft_config, training_config)
    
    # Train
    trainer = SFTTrainerWrapper(model, tokenizer, base_config, data_config, training_config)
    final_path = trainer.train()
    
    logger.info("=" * 80)
    logger.info(f"✓ SFT COMPLETE! Model saved to: {final_path}")
    logger.info("=" * 80)


@app.command()
def dpo(
    model_name: str = typer.Option("microsoft/Phi-3.5-mini-instruct", help="Base model or SFT checkpoint"),
    output_dir: str = typer.Option("./results_dpo", help="Output directory"),
    beta: float = typer.Option(0.1, help="DPO beta parameter"),
    learning_rate: float = typer.Option(5e-7, help="Learning rate for DPO"),
):
    """Run Direct Preference Optimization (reward-free alignment)."""
    
    check_gpu_availability()
    base_config = BaseConfig(model_name=model_name, output_dir=output_dir)
    set_seed(base_config.seed)
    
    data_config = DataConfig()
    dpo_config = DPOTrainingConfig(beta=beta, learning_rate=learning_rate)
    
    # Load model with LoRA for DPO
    peft_config = PEFTConfig(peft_method="lora")
    training_config = TrainingConfig()
    
    model, tokenizer = load_model_and_tokenizer(base_config, peft_config, training_config)
    
    # Train
    trainer = DPOTrainerWrapper(model, tokenizer, base_config, data_config, dpo_config)
    final_path = trainer.train()
    
    logger.info("=" * 80)
    logger.info(f"✓ DPO COMPLETE! Model saved to: {final_path}")
    logger.info("=" * 80)


@app.command()
def grpo(
    model_name: str = typer.Option("microsoft/Phi-3.5-mini-instruct", help="Base model"),
    output_dir: str = typer.Option("./results_grpo", help="Output directory"),
    learning_rate: float = typer.Option(5e-6, help="Learning rate"),
):
    """Run Group Relative Policy Optimization (DeepSeek-R1 style)."""
    
    check_gpu_availability()
    base_config = BaseConfig(model_name=model_name, output_dir=output_dir)
    set_seed(base_config.seed)
    
    data_config = DataConfig(use_reasoning_data=True)
    grpo_config = GRPOConfig(learning_rate=learning_rate)
    
    # Load model
    peft_config = PEFTConfig(peft_method="lora")
    training_config = TrainingConfig()
    model, tokenizer = load_model_and_tokenizer(base_config, peft_config, training_config)
    
    # Train
    trainer = GRPOTrainerWrapper(model, tokenizer, base_config, data_config, grpo_config)
    final_path = trainer.train()
    
    logger.info("=" * 80)
    logger.info(f"✓ GRPO COMPLETE! Model saved to: {final_path}")
    logger.info("=" * 80)


@app.command()
def pipeline(
    stages: str = typer.Option("sft,dpo", help="Comma-separated stages: sft,dpo,grpo"),
    model_name: str = typer.Option("microsoft/Phi-3.5-mini-instruct", help="Base model"),
    output_dir: str = typer.Option("./results_pipeline", help="Output directory"),
):
    """Run full training pipeline: SFT -> DPO -> GRPO."""
    
    logger.info("=" * 80)
    logger.info("STARTING FULL TRAINING PIPELINE")
    logger.info(f"Stages: {stages}")
    logger.info("=" * 80)
    
    stage_list = [s.strip() for s in stages.split(",")]
    current_model = model_name
    
    for stage in stage_list:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"PIPELINE STAGE: {stage.upper()}")
        logger.info(f"{'=' * 80}\n")
        
        if stage == "sft":
            # Run SFT
            check_gpu_availability()
            base_config = BaseConfig(model_name=current_model, output_dir=f"{output_dir}/sft")
            set_seed(base_config.seed)
            
            data_config = DataConfig(use_reasoning_data=True)
            peft_config = PEFTConfig(peft_method="lora")
            training_config = TrainingConfig(num_train_epochs=2)
            
            model, tokenizer = load_model_and_tokenizer(base_config, peft_config, training_config)
            trainer = SFTTrainerWrapper(model, tokenizer, base_config, data_config, training_config)
            current_model = trainer.train()
            
        elif stage == "dpo":
            # Run DPO
            base_config = BaseConfig(model_name=current_model, output_dir=f"{output_dir}/dpo")
            set_seed(base_config.seed)
            
            data_config = DataConfig()
            dpo_config = DPOTrainingConfig()
            peft_config = PEFTConfig(peft_method="lora")
            training_config = TrainingConfig()
            
            model, tokenizer = load_model_and_tokenizer(base_config, peft_config, training_config)
            trainer = DPOTrainerWrapper(model, tokenizer, base_config, data_config, dpo_config)
            current_model = trainer.train()
        
        elif stage == "grpo":
            # Run GRPO
            base_config = BaseConfig(model_name=current_model, output_dir=f"{output_dir}/grpo")
            set_seed(base_config.seed)
            
            data_config = DataConfig(use_reasoning_data=True)
            grpo_config = GRPOConfig()
            peft_config = PEFTConfig(peft_method="lora")
            training_config = TrainingConfig()
            
            model, tokenizer = load_model_and_tokenizer(base_config, peft_config, training_config)
            trainer = GRPOTrainerWrapper(model, tokenizer, base_config, data_config, grpo_config)
            current_model = trainer.train()
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ FULL PIPELINE COMPLETE!")
    logger.info(f"Final model: {current_model}")
    logger.info("=" * 80)

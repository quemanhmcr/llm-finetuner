# -*- coding: utf-8 -*-
"""Cấu hình huấn luyện cho hệ thống fine-tuning ML.

Module này chứa các lớp cấu hình cho các tham số huấn luyện,
bao gồm các thiết lập tối ưu hóa và hiệu suất.
"""

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Cấu hình huấn luyện với các tối ưu hóa hiệu suất.
    
    Attributes:
        num_train_epochs (int): Số lượng epochs huấn luyện.
        per_device_train_batch_size (int): Kích thước batch cho mỗi thiết bị.
        gradient_accumulation_steps (int): Gradient accumulation.
        learning_rate (float): Tốc độ học - cao hơn cho các phương pháp PEFT.
        lr_scheduler_type (str): Bộ lập lịch LR.
        warmup_ratio (float): Tỷ lệ warmup.
        optim (str): Optimizer - hiệu quả về bộ nhớ.
        max_grad_norm (float): Gradient clipping.
        bf16 (bool): Sử dụng bfloat16.
        fp16 (bool): Sử dụng fp16.
        gradient_checkpointing (bool): Gradient checkpointing - tiết kiệm bộ nhớ.
        logging_steps (int): Số bước logging.
        save_steps (int): Số bước lưu.
        eval_steps (int): Số bước đánh giá.
        save_total_limit (int): Giới hạn tổng số lần lưu.
    """
    num_train_epochs: int = field(default=3, metadata={"help": "Số lượng epochs"})
    
    # Batch sizes - optimized for 24GB GPU
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Kích thước batch cho mỗi thiết bị"})
    gradient_accumulation_steps: int = field(
        default=16,  # Effective batch size = 16
        metadata={"help": "Gradient accumulation"}
    )
    
    # Learning rates - critical for PEFT
    learning_rate: float = field(
        default=2e-4,  # Higher LR for PEFT (1e-4 to 5e-4 recommended)
        metadata={"help": "Tốc độ học - cao hơn cho các phương pháp PEFT"}
    )
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "Bộ lập lịch LR"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Tỷ lệ warmup"})
    
    # Optimization
    optim: str = field(default="paged_adamw_8bit", metadata={"help": "Optimizer - hiệu quả về bộ nhớ"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Gradient clipping"})
    
    # Precision
    bf16: bool = field(default=True, metadata={"help": "Sử dụng bfloat16"})
    fp16: bool = field(default=False)
    
    # Memory optimizations
    gradient_checkpointing: bool = field(
        default=True, 
        metadata={"help": "Gradient checkpointing - tiết kiệm bộ nhớ"}
    )
    
    # Logging
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    eval_steps: int = field(default=500)
    save_total_limit: int = field(default=3)

# -*- coding: utf-8 -*-
"""Cấu hình DPO cho hệ thống fine-tuning ML.

Module này chứa các lớp cấu hình cho Tối ưu hóa Ưu tiên Trực tiếp,
một phương pháp căn chỉnh không cần reward.
"""

from dataclasses import dataclass, field


@dataclass
class DPOTrainingConfig:
    """Cấu hình cụ thể cho DPO.
    
    Attributes:
        beta (float): Tham số beta DPO - kiểm soát sự phân kỳ KL.
        learning_rate (float): Tốc độ học cho DPO - thấp hơn theo khuyến nghị.
        num_train_epochs (int): Số lượng epochs huấn luyện.
        per_device_train_batch_size (int): Kích thước batch cho mỗi thiết bị.
        gradient_accumulation_steps (int): Gradient accumulation.
    """
    beta: float = field(
        default=0.1,
        metadata={"help": "Tham số beta DPO - kiểm soát sự phân kỳ KL"}
    )
    learning_rate: float = field(
        default=5e-7,  # Lower LR for DPO as recommended
        metadata={"help": "Tốc độ học cho DPO - thấp hơn theo khuyến nghị"}
    )
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)

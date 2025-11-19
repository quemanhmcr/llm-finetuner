# -*- coding: utf-8 -*-
"""Cấu hình GRPO cho hệ thống fine-tuning ML.

Module này chứa các lớp cấu hình cho Tối ưu hóa Chính sách Tương đối Nhóm,
một phương pháp RL nâng cao được lấy cảm hứng từ DeepSeek-R1.
"""

from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    """GRPO (Tối ưu hóa Chính sách Tương đối Nhóm) - phong cách DeepSeek-R1.
    
    Attributes:
        learning_rate (float): Tốc độ học cho GRPO.
        num_train_epochs (int): Số lượng epochs huấn luyện.
        kl_coeff (float): Hệ số phân kỳ KL.
        group_size (int): Số lượng phản hồi cho mỗi prompt.
    """
    learning_rate: float = field(default=5e-6)
    num_train_epochs: int = field(default=1)
    kl_coeff: float = field(default=0.1, metadata={"help": "Hệ số phân kỳ KL"})
    group_size: int = field(default=8, metadata={"help": "Số lượng phản hồi cho mỗi prompt"})

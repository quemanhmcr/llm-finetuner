# -*- coding: utf-8 -*-
"""Cấu hình dữ liệu cho hệ thống fine-tuning ML.

Module này chứa các lớp cấu hình cho xử lý dữ liệu,
bao gồm lựa chọn dataset và các tham số tiền xử lý.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Cấu hình dữ liệu tập trung vào các dataset suy luận.
    
    Attributes:
        use_reasoning_data (bool): Sử dụng dataset suy luận (GSM8K, MetaMathQA).
        max_seq_length (int): Độ dài chuỗi tối đa.
        train_size (int): Số lượng ví dụ huấn luyện (sử dụng -1 cho toàn bộ dataset).
        math_dataset (str): Dataset suy luận toán học.
        dpo_dataset (str): Dataset ưu tiên cho DPO.
        reasoning_datasets (List[str]): Danh sách các dataset suy luận để sử dụng.
    """
    use_reasoning_data: bool = field(
        default=True, 
        metadata={"help": "Sử dụng dataset suy luận (GSM8K, MetaMathQA)"}
    )
    max_seq_length: int = field(default=1024, metadata={"help": "Độ dài chuỗi tối đa"})
    train_size: int = field(
        default=10000, 
        metadata={"help": "Số lượng ví dụ huấn luyện (sử dụng -1 cho toàn bộ dataset)"}
    )
    
    # Dataset sources
    math_dataset: str = field(default="openai/gsm8k", metadata={"help": "Dataset suy luận toán học"})
    dpo_dataset: str = field(
        default="HuggingFaceH4/ultrafeedback_binarized",
        metadata={"help": "Dataset ưu tiên cho DPO"}
    )
    
    # Additional datasets for reasoning
    reasoning_datasets: List[str] = field(
        default_factory=lambda: [
            "openai/gsm8k",
            "meta-math/MetaMathQA",
        ],
        metadata={"help": "Danh sách các dataset suy luận để sử dụng"}
    )

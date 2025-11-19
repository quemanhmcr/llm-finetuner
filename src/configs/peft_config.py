# -*- coding: utf-8 -*-
"""Cấu hình PEFT cho hệ thống fine-tuning ML.

Module này chứa các lớp cấu hình cho các phương pháp Fine-Tuning Hiệu quả về Tham số
bao gồm LoRA, QLoRA, và DoRA.
"""

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class PEFTConfig:
    """Cấu hình PEFT với các tham số tối ưu dựa trên nghiên cứu.
    
    Attributes:
        peft_method (Literal): Phương pháp PEFT: lora, qlora, dora.
        lora_r (int): Rank LoRA - cao hơn cho các tác vụ suy luận phức tạp.
        lora_alpha (int): Alpha LoRA - nên gấp đôi rank.
        lora_dropout (float): Dropout LoRA.
        target_modules (List[str]): Các module mục tiêu để có kết quả tốt nhất.
        use_4bit (bool): Sử dụng quantization 4-bit.
        bnb_4bit_quant_type (str): Loại quantization 4-bit.
        bnb_4bit_compute_dtype (str): Kiểu dữ liệu tính toán 4-bit.
        use_double_quant (bool): Double quantization cho QLoRA.
    """
    peft_method: Literal["lora", "qlora", "dora"] = field(default="lora")
    
    # LoRA hyperparameters - optimized based on recent research
    lora_r: int = field(
        default=32,  # Increased from 16 - better for complex tasks
        metadata={"help": "Rank LoRA - cao hơn cho các tác vụ suy luận"}
    )
    lora_alpha: int = field(
        default=64,  # 2x rank as recommended
        metadata={"help": "Alpha LoRA - nên gấp đôi rank"}
    )
    lora_dropout: float = field(
        default=0.05,  # 5% for larger models, 10% for 7-13B
        metadata={"help": "Dropout LoRA"}
    )
    
    # Target modules - ALL linear layers for best adaptation quality
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Mục tiêu tất cả các lớp tuyến tính để có kết quả tốt nhất"}
    )
    
    # Quantization for QLoRA
    use_4bit: bool = field(default=True)
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_4bit_compute_dtype: str = field(default="bfloat16")
    use_double_quant: bool = field(default=True, metadata={"help": "Double quantization cho QLoRA"})

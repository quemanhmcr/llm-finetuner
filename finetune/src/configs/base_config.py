# -*- coding: utf-8 -*-
"""Cấu hình cơ bản cho hệ thống fine-tuning ML.

Module này chứa lớp cấu hình cơ bản cung cấp các thiết lập chung
được sử dụng trên tất cả các thành phần huấn luyện.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseConfig:
    """Cấu hình cơ bản với các giá trị mặc định hợp lý.
    
    Attributes:
        model_name (str): Tên mô hình cơ sở - Phi-3.5 được khuyến nghị cho suy luận.
        output_dir (str): Thư mục đầu ra cho mô hình đã huấn luyện.
        seed (int): Seed ngẫu nhiên để đảm bảo tính tái tạo.
        cache_dir (Optional[str]): Thư mục cache cho mô hình và dataset.
    """
    model_name: str = field(
        default="microsoft/Phi-3.5-mini-instruct",  # Better reasoning baseline
        metadata={"help": "Mô hình cơ sở - Phi-3.5 được khuyến nghị cho suy luận"}
    )
    output_dir: str = field(default="./model", metadata={"help": "Thư mục đầu ra"})
    seed: int = field(default=42, metadata={"help": "Seed ngẫu nhiên"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Thư mục cache cho mô hình/dataset"})

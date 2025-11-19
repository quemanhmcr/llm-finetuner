# LLM Reasoning Finetuner

Một framework thử nghiệm để tinh chỉnh (fine-tune) các mô hình ngôn ngữ lớn, tập trung vào khả năng suy luận (reasoning) và tối ưu hóa sở thích (preference optimization). Dự án này tích hợp các kỹ thuật hiện đại như LoRA/QLoRA, DPO và GRPO để phục vụ mục đích nghiên cứu và học tập.

## Tính năng chính

- **PEFT (Parameter-Efficient Fine-Tuning)**: Hỗ trợ LoRA, QLoRA, DoRA.
- **Preference Optimization**: Tích hợp DPO (Direct Preference Optimization) và GRPO (Group Relative Policy Optimization).
- **Reasoning Focus**: Hỗ trợ xử lý các dataset suy luận như GSM8K, MetaMathQA.
- **Performance**: Tích hợp Flash Attention 2, gradient checkpointing.
- **Đánh giá và giám sát toàn diện**

## Dựa trên Nghiên cứu Mới nhất

- **DeepSeek-R1 (2025)**: GRPO và căn chỉnh suy luận
- **Bài báo QLoRA**: Các phương pháp tốt nhất cho quantization 4-bit
- **DoRA**: Cải thiện phân rã trọng số
- **DPO**: Tối ưu hóa ưu tiên không cần reward

## Cài đặt

```bash
# Clone repository
git clone https://github.com/example/finetune.git
cd finetune

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt package
pip install -e .
```

## Bắt đầu Nhanh

### 1. SFT với LoRA tối ưu trên dataset suy luận

```bash
python -m finetune.src sft --peft-method lora --use-reasoning-data
```

### 2. QLoRA với quantization 4-bit (hiệu quả bộ nhớ)

```bash
python -m finetune.src sft --peft-method qlora --batch-size 2
```

### 3. DPO để căn chỉnh ưu tiên

```bash
python -m finetune.src dpo --beta 0.1
```

### 4. GRPO (phong cách DeepSeek) - tiên tiến nhất

```bash
python -m finetune.src grpo --learning-rate 5e-6
```

### 5. Pipeline đầy đủ: SFT -> DPO

```bash
python -m finetune.src pipeline --stages sft,dpo
```

## Cấu trúc Dự án

```
finetune/
├── src/
│   ├── configs/          # Các lớp cấu hình
│   ├── core/            # Chức năng cốt lõi
│   ├── data/            # Bộ xử lý dataset
│   ├── models/          # Tiện ích tải mô hình
│   ├── trainers/        # Triển khai huấn luyện
│   ├── utils/           # Các hàm tiện ích
│   └── cli/            # Giao diện dòng lệnh
├── tests/              # Kiểm thử đơn vị
├── docs/               # Tài liệu
├── scripts/            # Các kịch bản trợ giúp
└── configs/            # Các tệp cấu hình
```

## Cấu hình

Hệ thống sử dụng hệ thống cấu hình mô-đun với các lớp riêng biệt cho các khía cạnh khác nhau:

- `BaseConfig`: Các thiết lập chung
- `DataConfig`: Cấu hình dataset
- `PEFTConfig`: Các thiết lập fine-tuning hiệu quả về tham số
- `TrainingConfig`: Các siêu tham số huấn luyện
- `DPOTrainingConfig`: Các thiết lập riêng cho DPO
- `GRPOConfig`: Các thiết lập riêng cho GRPO

## Sử dụng Nâng cao

### Cấu hình Tùy chỉnh

```python
from finetune.src import (
    BaseConfig, DataConfig, PEFTConfig, TrainingConfig,
    load_model_and_tokenizer, SFTTrainerWrapper
)

# Tạo cấu hình tùy chỉnh
base_config = BaseConfig(model_name="microsoft/Phi-3.5-mini-instruct")
data_config = DataConfig(use_reasoning_data=True, max_seq_length=2048)
peft_config = PEFTConfig(peft_method="lora", lora_r=64)
training_config = TrainingConfig(
    num_train_epochs=5,
    learning_rate=1e-4,
    per_device_train_batch_size=2
)

# Tải mô hình và huấn luyện
model, tokenizer = load_model_and_tokenizer(base_config, peft_config, training_config)
trainer = SFTTrainerWrapper(model, tokenizer, base_config, data_config, training_config)
trainer.train()
```

## Mẹo Hiệu suất

1. **Sử dụng QLoRA để tiết kiệm bộ nhớ**: Quantization 4-bit giảm sử dụng bộ nhớ ~75%
2. **Bật Flash Attention 2**: Tăng tốc 20-30% cho các thao tác attention
3. **Gradient checkpointing**: Đổi tính toán lấy bộ nhớ để phù hợp với mô hình lớn hơn
4. **Độ chính xác hỗn hợp**: Sử dụng bfloat16 trên phần cứng được hỗ trợ để huấn luyện nhanh hơn

## Đóng góp

Chúng tôi chào đón các đóng góp! Vui lòng xem [Hướng dẫn Đóng góp](CONTRIBUTING.md) để biết chi tiết.

## Giấy phép

Dự án này được cấp phép theo Giấy phép MIT - xem tệp [LICENSE](LICENSE) để biết chi tiết.

## Trích dẫn

Nếu bạn sử dụng mã này trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@misc{finetune2025,
  title={Hệ thống Fine-tuning ML Chuyên nghiệp Nâng cao - Phiên bản 2025 Tối ưu},
  author={Đội ngũ Fine-tuning ML},
  year={2025},
  url={https://github.com/example/finetune}
}
```

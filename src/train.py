"""LoRA supervised fine-tuning entry point for Qwen3-8B."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable


DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
DEFAULT_OUTPUT_DIR = (
    "/root/autodl-tmp/outputs/llm4wulinlegend/qwen3_tongxiangyu_lora"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--train_file", default="data/processed/tongxiangyu_sft.jsonl"
    )
    parser.add_argument(
        "--system_prompt_file", default="configs/system_prompt.txt"
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--report_to",
        default="tensorboard",
        choices=("tensorboard", "none"),
    )
    return parser.parse_args()


def load_training_dataset(train_file: str) -> Any:
    from datasets import load_dataset

    return load_dataset(
        "json",
        data_files={"train": train_file},
        split="train",
    )


def read_system_prompt(path: str) -> str:
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"系统提示词文件不存在: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8", errors="strict").strip()
    if not prompt:
        raise ValueError("系统提示词不能为空")
    return prompt


def build_preprocess_function(
    tokenizer: Any, system_prompt: str, max_length: int
) -> Callable[[dict[str, str]], dict[str, list[int]]]:
    if max_length <= 1:
        raise ValueError("max_length 必须大于 1")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 必须定义 eos_token_id")

    def process_func(example: dict[str, str]) -> dict[str, list[int]]:
        if example.get("input") != "":
            raise ValueError("当前版本要求 input 严格为空字符串")

        user_content = example["instruction"]
        answer = example["output"]
        if not user_content.strip() or not answer.strip():
            raise ValueError("instruction 和 output 均不能为空")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        answer_ids = tokenizer(
            answer,
            add_special_tokens=False,
        )["input_ids"] + [tokenizer.eos_token_id]

        available_prompt_length = max_length - len(answer_ids)
        if available_prompt_length <= 0:
            raise ValueError("assistant answer exceeds max_length")

        # Truncate the start of the prompt but always retain the full answer.
        prompt_ids = prompt_ids[-available_prompt_length:]
        input_ids = prompt_ids + answer_ids

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + answer_ids,
        }

    return process_func


def build_model_and_tokenizer(model_name_or_path: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 CUDA GPU。Qwen3-8B LoRA 训练前请在 AutoDL 挂载 GPU。"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16，请更换支持 BF16 的 GPU")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    return model, tokenizer


def build_lora_model(
    model: Any,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()
    return lora_model


def validate_paths(args: argparse.Namespace) -> None:
    model_path = Path(args.model_name_or_path)
    train_path = Path(args.train_file)
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"模型目录缺少 config.json: {model_path}")
    if not train_path.is_file():
        raise FileNotFoundError(f"训练数据不存在: {train_path}")


def main() -> None:
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    args = parse_args()
    validate_paths(args)
    system_prompt = read_system_prompt(args.system_prompt_file)

    model, tokenizer = build_model_and_tokenizer(args.model_name_or_path)
    model = build_lora_model(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    dataset = load_training_dataset(args.train_file)
    if len(dataset) == 0:
        raise ValueError("训练数据集为空")

    process_func = build_preprocess_function(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        max_length=args.max_length,
    )
    tokenized_dataset = dataset.map(
        process_func,
        remove_columns=dataset.column_names,
        desc="Tokenizing training dataset",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        bf16=True,
        report_to=[] if args.report_to == "none" else [args.report_to],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
        ),
    )

    torch.manual_seed(args.seed)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA 适配器已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

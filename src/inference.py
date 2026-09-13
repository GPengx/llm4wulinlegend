"""交互式体验 Qwen3-8B 或训练后的佟湘玉 LoRA 模型。"""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"
ADAPTER_PATH = (
    "/root/autodl-tmp/outputs/llm4wulinlegend/qwen3_tongxiangyu_lora"
)
SYSTEM_PROMPT_PATH = "configs/system_prompt.txt"


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

if (Path(ADAPTER_PATH) / "adapter_config.json").is_file():
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print(f"已加载 LoRA：{ADAPTER_PATH}")
else:
    print("未找到 LoRA，当前使用基础模型。")

model.eval()
system_prompt = Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8").strip()

print("输入问题开始对话，输入 exit 退出。")
while True:
    user_text = input("\n你：").strip()
    if user_text.lower() in {"exit", "quit"}:
        break
    if not user_text:
        continue

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    answer = tokenizer.decode(
        outputs[0][model_inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    print(f"佟湘玉：{answer}")

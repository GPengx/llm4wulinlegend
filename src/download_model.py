"""下载 Qwen3-8B 模型。"""

from modelscope import snapshot_download


model_dir = snapshot_download(
    "Qwen/Qwen3-8B",
    local_dir="/root/autodl-tmp/models/Qwen3-8B",
    revision="master",
)

print(f"模型下载完成，保存位置：{model_dir}")

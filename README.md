![llm4武林外传：同福客栈漫画主题横幅](assets/tongfu-inn-banner-v2.png)

<h1 align="center">LLM4WulinLegend · 同福客栈，开聊啦</h1>

<h2 align="center">把江湖烟火气，带进大模型对话里。</h2>

<h3 align="center">Qwen3-8B · LoRA 微调 · 佟湘玉角色对话</h3>

使用 LoRA 对 Qwen3-8B 进行单轮监督微调，使模型学习佟湘玉风格回答。项目包含原文编码规范化、剧本台词抽取、单轮问答构造、数据校验和训练入口。

从一段剧本台词到一次角色对话，跟着项目体验数据处理、模型下载、微调与推理的完整流程。

> 本项目仅用于学习与研究，与原著作者、版权方及模型提供方无关。仓库许可证只覆盖本项目原创代码，不授权使用小说原文、影视角色、第三方模型或其权重。

## 🚀 快速开始

以下命令在 AutoDL 服务器终端执行。使用前需安装项目依赖和基础模型，并挂载支持 BF16 的 GPU；首次使用请先参照下方“1. 安装依赖”和“2. 下载模型”。

克隆项目并进入项目目录：

```bash
# 1. 克隆项目
cd /root
git clone https://github.com/GuanPengxin/llm4wulinlegend.git
cd /root/llm4wulinlegend

# 2. 启动推理，体验对话
python src/inference.py
```

脚本检测到最终 LoRA 时会自动加载微调结果；否则使用基础模型配合角色提示词回答。仓库不附带模型权重，体验微调效果需先按“4. 训练”生成 LoRA。

推理时输入问题，例如“掌柜的，今天生意不好怎么办？”，输入 `exit` 退出。当前每次提问独立处理，不保存聊天历史。

后续命令均在项目根目录 `/root/llm4wulinlegend` 下执行。

## 目录与数据边界

项目代码位于 `/root/llm4wulinlegend`，基础模型和训练输出使用 AutoDL 数据盘：

```text
/root/autodl-tmp/models/Qwen3-8B
/root/autodl-tmp/outputs/llm4wulinlegend
```

## 1. 安装依赖

```bash
cd /root/llm4wulinlegend
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. 下载模型

```bash
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/outputs/llm4wulinlegend

python src/download_model.py
```

脚本通过 ModelScope 下载 `Qwen/Qwen3-8B` 的 `master` 版本，并直接保存到
`/root/autodl-tmp/models/Qwen3-8B`。如需更换模型或目录，直接修改
[`src/download_model.py`](src/download_model.py) 中对应的参数即可。网络中断后再次执行
同一命令，可以复用已经下载的文件。

## 3. 准备单轮数据

仓库附带最终单轮 SFT 训练集 `data/processed/tongxiangyu_sft.jsonl`，当前共 `5218` 条。直接使用时，只需校验，无需重新处理原文：

```bash
python scripts/validate_dataset.py --input data/processed/tongxiangyu_sft.jsonl
```

### 数据目录与格式

- `data/samples/sample_sft.jsonl`：虚构的格式示例。
- `data/interim/`：编码规范化及台词抽取的中间结果，本地生成且不提交。
- `data/processed/tongxiangyu_sft.jsonl`：随仓库提供的完整训练集。

最终数据使用 UTF-8 JSONL 格式，每行是一个 JSON 对象。以下为虚构示例：

```json
{"instruction":"掌柜的，饭凉了怎么办？","input":"","output":"拿去热一下，别浪费了。"}
```

`instruction` 只包含紧邻佟湘玉回答之前的一条其他人物台词正文，不添加说话人前缀；`input` 必须为空字符串 `""`；`output` 为佟湘玉回答，不包含说话人前缀。当前不拼接多轮历史。

项目提供数据处理代码、格式说明、虚构样例及最终训练集；小说或剧本文本和中间抽取数据不公开。使用者应自行确认数据来源、授权范围和适用法律。仓库的代码许可证不自动授予小说文本、影视角色、模型权重或训练数据的版权许可。

### 自行从原文构建（可选）

如需重新生成数据，将原文放到仓库根目录并命名为 `wulinlegnd.txt`，依次运行以下命令。此操作会覆盖仓库附带的最终训练集：

```bash
python scripts/normalize_text.py \
  --input wulinlegnd.txt \
  --output data/interim/wulinlegnd_utf8.txt

python scripts/extract_dialogues.py \
  --input data/interim/wulinlegnd_utf8.txt \
  --output data/interim/dialogues.jsonl

python scripts/build_dataset.py \
  --input data/interim/dialogues.jsonl \
  --output data/processed/tongxiangyu_sft.jsonl

python scripts/validate_dataset.py \
  --input data/processed/tongxiangyu_sft.jsonl
```

自动抽取不能替代人工审核。正式训练前，请抽查至少 200～500 条样本，确认说话人和相邻关系正确。
为保持训练与普通用户推理时的输入格式一致，最终 `instruction` 只保存上一句台词正文，
不包含“白展堂：”等说话人前缀。

## 4. 训练

```bash
python src/train.py \
  --model_name_or_path /root/autodl-tmp/models/Qwen3-8B \
  --train_file data/processed/tongxiangyu_sft.jsonl \
  --system_prompt_file configs/system_prompt.txt \
  --output_dir /root/autodl-tmp/outputs/llm4wulinlegend/qwen3_tongxiangyu_lora \
  --max_length 512 \
  --warmup_ratio 0.03
```

训练代码使用 Qwen3 自带聊天模板并关闭 thinking；system/user token 的标签为 `-100`，损失只作用于 assistant 回答。
学习率会先在总训练步数的前 `3%` 线性升至 `1e-4`，随后线性衰减至接近 `0`。

## 5. 快速推理

训练完成后运行交互式推理脚本：

```bash
python src/inference.py
```

输入问题即可对话，输入 `exit` 退出。脚本会优先加载
`/root/autodl-tmp/outputs/llm4wulinlegend/qwen3_tongxiangyu_lora` 中的 LoRA；
如果尚未训练或未找到 LoRA，则使用 Qwen3-8B 基础模型。

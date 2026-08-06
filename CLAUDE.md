# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Windows 环境注意**：本机为 Windows。命令行操作应使用 PowerShell。Git Bash 中不存在的命令（如 `bash`、`docker`）需通过 `powershell -Command "..."` 调用，或在 README 中明确按 Unix 写法时改为 PowerShell 等价命令。

---

## 项目概览

本仓库基于 Gaëtan Hadjeres 等人在 ICML 2017 发表的论文 *DeepBach: a Steerable Model for Bach Chorales Generation*，并在其 PyTorch 复现基础上做了面向**中文简谱（jianshi）单声部音乐**的二次开发与扩展。

主要代码来自原 Ghadjeres/DeepBach 仓库；中文简谱相关脚本（`preprocess_chinese_score*.py`、`train_simple_notation.py`、`generate_simple_notation.py`、`generate_constrained.py`、`generate_multi_segment.py`、`json_to_midi.py`、`midi_to_json.py`）为本仓库新增。

参考资料：
- 论文：<http://proceedings.mlr.press/v70/hadjeres17a.html>
- Keras 原版：`original_keras` 分支

---

## 安装与环境

### 原版 DeepBach 环境

```powershell
conda env create --name deepbach_pytorch -f environment.yml
bash dl_dataset_and_models.sh      # 下载预训练模型 + Bach 赞美诗数据集（约几十 MB）
```

`environment.yml` 固定 Python 3.6 + PyTorch 1.0 + music21 5.5.0 + Flask 1.0.2。

### 中文简谱扩展环境

按 `简谱三脚本执行说明.md` 推荐：

```powershell
$PY = "D:\code\music\deepbach_learn\code\.venv\Scripts\python.exe"
& $PY -m pip install music21 torch click tqdm numpy matplotlib
```

`music21` 需要外部 musicxml 渲染器才能显示乐谱：
- 有 GUI：`python -c "import music21; music21.environment.set('musicxmlPath', '/path/to/musescore')"`
- 无 GUI / 服务器：`python -c "import music21; music21.environment.set('musicxmlPath', '/bin/true')"`

---

## 常用命令

> 所有 `python` 在文档中默认替换为项目 venv 的解释器 `$PY`。

### 完整工作流（中文简谱）

按 `简谱三脚本执行说明.md`，简谱生成是三步流水线：JSON → PT → 训练 → 生成。

#### 1. 预处理（JSON → PT）

```powershell
& $PY .\preprocess_chinese_score_batch.py `
  --input    "D:/code/music/sizhu_data/data/tang" `
  --output   ".\preprocessed_data" `
  --sequence_size 8 `
  --subdivision  8
```

- `--input`：JSON 文件夹或单文件。
- `--subdivision`：默认 8（支持 32 分音符精度）；subdivision=4 会丢失 0.125 拍粒度。
- 输出：`preprocessed_data/dataset_<suffix>_combined_tensor_dataset.pt`

单文件处理：`preprocess_chinese_score.py --input <json> --output <dir> [--verify]`

#### 2. 训练

```powershell
& $PY .\train_simple_notation.py `
  --data   ".\preprocessed_data\dataset_tang_combined_tensor_dataset.pt" `
  --subdivision 8 `
  --note_embedding_dim 50 --meta_embedding_dim 25 `
  --num_layers 2 --lstm_hidden_size 128 `
  --linear_hidden_size 128 --dropout_lstm 0.5 `
  --train 30 --batch_size 16 `
  --models_dir ".\models"
```

模型文件名按参数编码：
```
voicemodel_{suffix}_ne{note_emb}_me{meta_emb}_lh{lstm}_ll{layers}_ld{dropout}_li{linear}_{voice_index}.pt
```
例：`voicemodel_tang_combined_tensor_dataset_ne50_me25_lh128_ll2_ld0.5_li128_0.pt`

> **关键约束**：`preprocess / train / generate` 三者使用的 `--subdivision` 必须一致。

#### 3. 生成

无约束（从头）：
```powershell
& $PY .\generate_simple_notation.py `
  --data  ".\preprocessed_data\dataset_tang_combined_tensor_dataset.pt" `
  --load --output generated.mid `
  --sequence_length_ticks 512 --num_iterations 500
```

带板眼约束（按 JSON 中 `ban=1` / `yan=1` 锚定）：
```powershell
& $PY .\generate_constrained.py `
  --json  "D:/code/music/sizhu_data/data/tang/汤·云庆.json" `
  --data  ".\preprocessed_data\dataset_tang_combined_tensor_dataset.pt" `
  --model ".\models\voicemodel_tang_combined_tensor_dataset_ne50_me25_lh128_ll2_ld0.5_li128_0.pt" `
  --output ".\generated_scores\tang_yunqing.mid" `
  --subdivision 8 --num_iterations 1000 `
  --temperature 1.0 --batch_size 8 --png
```

多段级联（每段间用 `ban=1` 划分并共享上下文）：
```powershell
& $PY .\generate_multi_segment.py `
  --json  ".\zhonghua-head.json" `
  --data  ".\preprocessed_data\dataset_jin_combined_tensor_dataset.pt" `
  --output generated_multi.mid
```

> `generate_constrained.py` 与 `generate_multi_segment.py` **必须** 用 `--model` 指定模型文件（参数从文件名解析）。命令行不再接受 `note_embedding_dim` / `lstm_hidden_size` 等。

### 原版 Bach 赞美诗工作流

```powershell
# 训练 + 生成
& $PY .\deepBach.py --train --num_epochs 5
# 加载预训练模型并生成
& $PY .\deepBach.py --num_iterations 500 --sequence_length_ticks 64
```

参数：见 `deepBach.py` Click 选项（`--note_embedding_dim 20`、`--lstm_hidden_size 256`、`--num_layers 2`、`--dropout_lstm 0.5`、`--batch_size 256`、`--num_epochs 5` 等默认）。

数据预缓存：`process_data.py` 在 `DatasetManager/dataset_cache/` 下生成 `tensor_datasets/` 与 `datasets/` 两类缓存。`DatasetManager.dataset_manager.DatasetManager.load_if_exists_or_initialize_and_save` 会优先读取缓存。

### Flask 服务器（NONOTO / MuseScore 集成）

```powershell
& $PY .\flask_server.py --port 5000
& $PY .\musescore_flask_server.py          # 旧版 MuseScore 插件后端
```

Docker：`docker run --runtime=nvidia -p 5000:5000 -it --rm ghadjeres/deepbach`

---

## 代码架构

### 目录结构

```
DeepBach/
├── DeepBach/                       # 核心 PyTorch 模型
│   ├── model_manager.py            # DeepBach 类：并行多声部、pseudo-Gibbs 生成
│   ├── voice_model.py              # VoiceModel：双向 LSTM + MLP，每声部一个
│   ├── metadata.py                 # 内置 metadata 定义（旧版，可参考）
│   ├── helpers.py                  # cuda_variable / to_numpy / init_hidden
│   └── data_utils.py               # mask_entry / reverse_tensor
├── DatasetManager/
│   ├── dataset_manager.py          # DatasetManager：缓存 + 注册 (bach_chorales 等)
│   ├── chorale_dataset.py          # ChoraleDataset / ChoraleBeatsDataset：基于 music21 的 Bach 数据集
│   ├── music_dataset.py            # MusicDataset 抽象基类（ABC）
│   ├── metadata.py                 # TickMetadata / FermataMetadata / KeyMetadata / ModeMetadata
│   └── helpers.py                  # SLUR_SYMBOL / START_SYMBOL / END_SYMBOL / REST_SYMBOL / PAD_SYMBOL
├── deepBach.py                     # 原版入口：训练或加载并生成 Bach 赞美诗
├── flask_server.py                 # NONOTO 集成 Flask 服务器（/generate, /timerange-change）
├── musescore_flask_server.py       # MuseScore 插件后端（已弃用）
├── predict.py                      # 简易预测脚本
├── inspect_cache.py                # 检查 DatasetManager 缓存
├── verify_dataset.py               # 数据集完整性校验
│
├── train_simple_notation.py        # 简谱训练脚本（封装 SimpleNotationDataset + DeepBach）
├── preprocess_chinese_score.py     # 单文件预处理
├── preprocess_chinese_score_batch.py  # 批量预处理（合并全局 note2index）
├── generate_simple_notation.py     # 简谱无约束生成（参数手动指定）
├── generate_constrained.py         # 简谱带 ban/yan 锚点的约束生成
├── generate_multi_segment.py       # 简谱多段级联生成（基于 ban=1 切段）
├── json_to_midi.py                 # 简谱 JSON → MIDI
├── midi_to_json.py                 # MIDI → 简谱 JSON（反向工具）
├── json_to_midi.py / midi_to_json.py
│
├── deepBachMuseScore.qml           # MuseScore 插件定义（已弃用）
├── process_data.py                 # 触发 DatasetManager 缓存重建
├── dataset_cache/                  # 缓存目录（被 .gitignore）
├── models/                         # 模型权重（被 .gitignore）
└── 运行脚本.md / 简谱三脚本执行说明.md / preprocessREADME.md   # 中文使用文档
```

### 核心数据流

```
music21 Score → MusicDataset.get_score_tensor + get_metadata_tensor
              → chorale_tensor (N, num_voices, ticks) + metadata_tensor (N, num_voices, ticks, num_metadata)
              → TensorDataset → DataLoader
              → VoiceModel.forward(notes, metas) → logits per voice
              → parallel_gibbs 采样更新 → tensor_chorale
              → MusicDataset.tensor_to_score → music21 Stream → MIDI / MusicXML / PDF
```

### DeepBach 模型核心

- **DeepBach** (`DeepBach/model_manager.py`)：顶层封装。每条声部独立训练一个 `VoiceModel`（论文 Fig.4 的并行结构）。
- **VoiceModel** (`DeepBach/voice_model.py`)：对单声部建模。架构：
  - `note_embeddings` / `meta_embeddings`：按声部/类别 Embedding。
  - `lstm_left` / `lstm_right`：双向 LSTM（同一时间步 t，左右各看 `timesteps_ticks` 个 tick）。
  - `mlp_center`：处理中间时刻其它声部的音符（自身声部被 mask）。
  - `mlp_predictions`：拼接 [left, center, right] → 主声部在该时刻的 logits。
  - `preprocess_input`：随机偏移 offset（避免总是训练正中央位置），把 tensor 拆成 (left, center, right, label)。
- **parallel_gibbs**：生成阶段执行 pseudo-Gibbs 采样。每轮：
  1. 温度退火：`temperature *= 0.9993`，下限 `--temperature`。
  2. 每条声部采 `batch_size_per_voice` 个时间步，对每个时间步送入对应 `VoiceModel` 得到 softmax 分布，按 `np.random.multinomial` 采样更新。
  3. 支持 `time_index_range_ticks` / `time_index_list_ticks` / `voice_index_range` 做局部重生成。
- 模型文件名编码格式见 `voice_model.parse_model_filename`，是 `generate_constrained.py` 自动恢复架构的关键。

### 数据集 & Metadata

- `MusicDataset`（ABC）定义 `get_score_tensor / get_metadata_tensor / tensor_to_score / extract_*_with_padding / empty_score_tensor / random_score_tensor` 等接口；`ChoraleDataset`（基于 music21 corpus）和 `ChoraleBeatsDataset` 是 Bach 赞美诗实现。
- `DatasetManager` 维护 `all_datasets` 注册表与 `dataset_cache/` 缓存。
- 简谱扩展 `SimpleNotationDataset`（在 `train_simple_notation.py`）绕过完整 `ChoraleDataset`，直接从 `.pt` 读取 `(chorale_tensor, metadata_tensor, note2index, index2note)` 并补齐 DeepBach 期望的接口（`data_loaders / extract_*_with_padding / tensor_to_score` 等）。
- `metadata_tensor` 列顺序约定（简谱版）：`[IsPlaying, Tick, Mode, Key, Fermata, voice_id]`；训练时只取 `[Tick, Mode, Key, Fermata, voice_id]`（跳过 IsPlaying）。

### 简谱扩展（论文未覆盖）

- **预处理**：`preprocess_chinese_score*.py` 把 JSON 中 `value ∈ {1..7, 0}, octave, duration` 转 MIDI pitch（`base + octave*12`，`{1→60, 2→62, ..., 7→71}`），展开为 tick 级序列（subdivision 控制精度）。滑动窗口采样后保存为 `.pt`。
- **板眼约束**：`generate_constrained.py` 把 JSON 中 `ban=1` / `yan=1` / 整数拍起点当作固定点。固定音位信息通过 `time_index_list_ticks` 传给 `parallel_gibbs`（从 allowed_positions 中排除）。
- **多段级联**：`generate_multi_segment.py` 按 ban 切段，每段作为邻段的上下文（共享同一 chorale_tensor），保证长曲连贯。
- **PNG 预览**：`generate_constrained.py --png` 通过 matplotlib 在无 GUI 环境（`matplotlib.use('Agg')`）绘制简谱。
- **模型文件名解析**：`DeepBach.voice_model.parse_model_filename` 从文件名恢复所有架构参数；`generate_*.py` 不再要求用户重传。

---

## 测试 / 验证 / 调试

仓库**没有**单元测试或 CI。校验工具：
- `verify_dataset.py`：检查数据张量形状、metadata 列范围。
- `inspect_cache.py`：枚举 `DatasetManager/dataset_cache/` 内容。
- `preprocess_chinese_score.py --verify`：打印前 20 个样本做人工对比。

训练中打印 train/val loss & acc（`VoiceModel.train_model`），并每轮 `self.save()`。

---

## 关键约定 / 易踩坑

1. **`subdivision` 全链路一致**：preprocess / train / generate 必须相同（推荐 8）。
2. **CUDA 自动检测**：`DeepBach.helpers.cuda_variable` 在无 CUDA 时跑 CPU；`model.cuda()` 只在 `torch.cuda.is_available()` 时生效。
3. **music21 渲染器**：缺省会抛 `SubConverterException`；服务器上设成 `/bin/true`。
4. **模型文件名**：`generate_constrained.py` 严格依赖 `voicemodel_<suffix>_ne..._lh..._ll..._ld..._li..._<voice>.pt` 格式，乱改名会导致解析失败。
5. **`time_index_range_ticks` 与 padding**：`DeepBach.generation` 在内部把所有 tick 坐标平移 `+timesteps_ticks`（左 padding 是 START_SYMBOL）。传 `time_index_range_ticks` 时使用**未平移的原始坐标**；`time_index_list_ticks` 同理。
6. **`ChoraleDataset.voice_ids` 未使用**（代码注释自承 TODO）：实际声部索引始终是 `[0, 1, 2, 3]`。
7. **生成是伪随机**：`parallel_gibbs` 用 `np.random` + `np.random.multinomial`，没有设 seed，要可复现需自行 `np.random.seed`。
8. **`flask_server.py` 端点**：`/generate`（无约束）、`/timerange-change`（带 `time_range_start_quarter` / `time_range_end_quarter` query + `sheet` / `fermatas` JSON body）、`/musicxml-to-midi`、`/test-generate`。
9. **Docker**：`Dockerfile` 基于 `pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime`，已弃；现在推荐直接 conda/venv。
10. **LICENSE**：GPL（见 `LICENSE` 文件）。

---

## 主要文档索引

- `README.md`：英文原版 DeepBach 入口（Bach 赞美诗 + Flask）。
- `运行脚本.md`：原始中文流程说明。
- `简谱三脚本执行说明.md`：中文简谱三步流水线（**最权威**，所有命令以 PowerShell 形式给出）。
- `preprocessREADME.md`：JSON ↔ DeepBach tensor 的映射规范。
- `运行脚本.md` 与 `preprocessREADME.md` 中存在历史命令（部分已迁移到 `generate_constrained.py --model` 用法，按 *简谱三脚本执行说明.md* 为准）。
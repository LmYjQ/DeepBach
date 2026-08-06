"""
简谱"整拍token"预处理脚本
=================================

新预处理方式：把"一整拍 duration=1.0 内"的所有连续音符（可能是单个全拍，也可能是
若干 0.5 / 0.25 / ... 时值的细音符）整体编码成**一个 token**。这样：

    - 一个 token = 一个完整的 beat
    - 同一个 beat 内的所有细音符被合并为一个离散的 beat-pattern 类别
    - 语料由"按 tick 展开"变成"按 beat 推进"，序列长度从 `sequence_size*subdivision`
      缩短到 `sequence_size`，模型需要学习的步数更少、视野更宽。

输入：JSON 简谱（与 `preprocess_chinese_score.py` 相同）
    - `value`: "1"-"7" / "0"(休止) / "bar"(过滤)
    - `octave`: 八度偏移
    - `duration`: 时值（拍，1.0 = 整拍）
    - `ban` / `yan`: 用于约束生成，本脚本预处理时不使用

输出 (与 `preprocess_chinese_score.py` 同结构，shape 不同)：
    - chorale_tensor : (N, 1, sequence_size)         每个元素是 beat-pattern 的索引
    - metadata_tensor: (N, 1, sequence_size, 6)
        列顺序与现有流水线保持一致：[IsPlaying, Tick, Mode, Key, Fermata, voice_id]
        Tick 全部填 0（一个 token 已经覆盖整拍，没有"拍内位置"概念）
    - beat2index / index2beat : beat-pattern ↔ 索引 的双向映射

用法:
    python preprocess_beat_token.py --input <json_path> --output <output_dir>
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch


# ---------- 工具 ----------

def note_to_pitch(value, octave):
    """
    简谱数字 + 八度 → MIDI pitch
    与 `json_to_midi.py` / `preprocess_chinese_score.py` 保持一致：
        value=1 -> C4=60, value=2 -> D4=62, ..., value=7 -> B4=71
        midi = base + 12 * octave
    """
    base = {1: 60, 2: 62, 3: 64, 4: 65, 5: 67, 6: 69, 7: 71}
    return base[int(value)] + 12 * int(octave)


def note_signature(note):
    """
    单个音符的规范化字符串形式。用于构造 beat-pattern。

    例: value=6 octave=0 duration=0.5  -> "67@0.5"
        value=0 (休止符)               -> "R@0.5"
    """
    value = note.get('value')
    if value is None or str(value).lower() in ('bar', 'space'):
        return None  # 过滤
    if str(value) == '0':
        return f"R@{note['duration']:g}"
    pitch = note_to_pitch(value, note.get('octave', 0))
    return f"{pitch}@{note['duration']:g}"


def notes_to_beats(notes, eps=1e-6):
    """
    把音符流切分为整拍组：每组累加 duration ≈ 1.0。

    返回: list[list[note_dict]]
        每个内层 list 是一整拍内的所有细音符。
    """
    beats = []
    current = []
    acc = 0.0
    for note in notes:
        sig = note_signature(note)
        if sig is None:
            continue  # 跳过小节线等
        current.append({**note, '_sig': sig, '_pitch': note_to_pitch(note['value'], note.get('octave', 0))
                                                          if str(note['value']) != '0' else 0})
        acc += float(note['duration'])
        if acc >= 1.0 - eps:
            beats.append(current)
            current = []
            acc = 0.0
    # 末尾不足 1 拍的零头单独成一拍（少见，但保险）
    if current:
        beats.append(current)
    return beats


def beat_to_pattern(beat_notes):
    """
    把一拍内的音符序列编码成一个 pattern 字符串。
    例: [全拍6]                  -> "67@1"
        [半拍1 + 半拍2]          -> "65@0.5|62@0.5"
        [半拍0 + 四分2 + 四分3]  -> "R@0.5|62@0.25|64@0.25"
    """
    return "|".join(n['_sig'] for n in beat_notes)


# ---------- 主流程 ----------

def preprocess(json_path, sequence_size=8, subdivision=8, verbose=True):
    """
    Args:
        json_path:     输入 JSON 文件
        sequence_size: 每样本多少拍
        subdivision:   保留参数，仅作日志（不影响 beat-token 输出）
        verbose:       是否打印

    Returns:
        dict: {chorale_tensor, metadata_tensor, beat2index, index2beat, ...}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    title = data.get('title', 'unknown')
    raw_notes = data.get('notes', [])
    beats = notes_to_beats(raw_notes)

    # 统计 beat pattern 频次
    pattern_counter = Counter(beat_to_pattern(b) for b in beats)

    if verbose:
        print('=' * 60)
        print(f'[整拍token预处理] {title}')
        print('=' * 60)
        print(f'原始音符数     : {len(raw_notes)}')
        print(f'拍数           : {len(beats)}')
        print(f'唯一 pattern 数: {len(pattern_counter)}')
        print()
        print('Beat-pattern 频次 (top 20):')
        for pat, cnt in pattern_counter.most_common(20):
            print(f'  {cnt:4d} ×  {pat}')
        print()

    # 构建词汇表：索引 0,1 预留给 START_SYMBOL / END_SYMBOL (供 padding 用)，
    # 后续按频次降序排 pattern，确保可复现。
    index2beat = {
        0: 'START',   # 与 DatasetManager.helpers.START_SYMBOL 保持一致
        1: 'END',     # 与 DatasetManager.helpers.END_SYMBOL   保持一致
    }
    sorted_patterns = sorted(pattern_counter.items(), key=lambda x: (-x[1], x[0]))
    for pat, _ in sorted_patterns:
        if pat in ('START', 'END'):  # 极端情况下 pattern 字符串撞名特殊符号的兜底
            continue
        index2beat[len(index2beat)] = pat
    beat2index = {pat: i for i, pat in index2beat.items()}

    if verbose:
        print(f'词表大小: {len(beat2index)} (含 START/END)')
        print('词表示例:')
        for i, (pat, idx) in enumerate(list(beat2index.items())[:10]):
            print(f'  [{idx}] -> {pat}')
        if len(beat2index) > 10:
            print(f'  ... 共 {len(beat2index)} 个')
        print()

    # 把每拍转成 token id
    beat_ids = np.array([beat2index[beat_to_pattern(b)] for b in beats], dtype=np.int64)

    # 拍级元数据 (整拍粒度)
    MODE_MAJOR = 1
    KEY_D_MAJOR = 10  # 2 sharps
    FERMATA_NONE = 0
    is_playing = np.array(
        [1 if any(str(n['value']) != '0' for n in b) else 0 for b in beats],
        dtype=np.int64,
    )
    tick_pos = np.zeros(len(beats), dtype=np.int64)        # 一拍一个 token，填 0
    mode_arr = np.full(len(beats), MODE_MAJOR, dtype=np.int64)
    key_arr = np.full(len(beats), KEY_D_MAJOR, dtype=np.int64)
    fermata_arr = np.zeros(len(beats), dtype=np.int64)
    voice_arr = np.zeros(len(beats), dtype=np.int64)

    # 滑窗采样：每 `sequence_size` 拍一个样本
    total_beats = len(beats)
    stride = 1  # 每拍一步
    chorale_list, meta_list = [], []
    for start in range(0, total_beats - sequence_size + 1, stride):
        end = start + sequence_size
        win_ids = beat_ids[start:end]                          # (sequence_size,)
        # 6 列 metadata，顺序与现有流水线一致
        win_meta = np.stack([
            is_playing[start:end],
            tick_pos[start:end],
            mode_arr[start:end],
            key_arr[start:end],
            fermata_arr[start:end],
            voice_arr[start:end],
        ], axis=-1)                                            # (sequence_size, 6)
        chorale_list.append(win_ids[np.newaxis, :])            # (1, sequence_size)
        meta_list.append(win_meta[np.newaxis, :, :])           # (1, sequence_size, 6)

    chorale_tensor = np.stack(chorale_list) if chorale_list else np.zeros((0, 1, sequence_size), dtype=np.int64)
    metadata_tensor = np.stack(meta_list) if meta_list else np.zeros((0, 1, sequence_size, 6), dtype=np.int64)

    if verbose:
        print('-' * 60)
        print('输出 tensor 形状:')
        print(f'  chorale_tensor : {chorale_tensor.shape}  dtype={chorale_tensor.dtype}')
        print(f'  metadata_tensor: {metadata_tensor.shape}  dtype={metadata_tensor.dtype}')
        print(f'  样本数         : {chorale_tensor.shape[0]}')
        print()
        print('前 3 个样本 (一拍一个整数):')
        for i in range(min(3, len(chorale_list))):
            tokens = chorale_list[i][0]
            decoded = [index2beat[t] for t in tokens]
            print(f'  sample {i}: {tokens.tolist()}')
            for j, dec in enumerate(decoded):
                print(f'          beat {j}: {dec}')
        print()

        # 对比：原 tick 级方案样本量
        if subdivision:
            approx_tick_samples = max(0, (total_beats * subdivision) - sequence_size * subdivision + 1)
            print(f'对照: tick 级方案 (`subdivision={subdivision}`) 同样序列长度会产生约 '
                  f'{approx_tick_samples} 个样本。')
            print(f'      beat-token 方案产生 {chorale_tensor.shape[0]} 个样本, '
                  f'每个样本等效于 {subdivision}× 更长的 tick 视野。')
            print()

    return {
        'chorale_tensor':   chorale_tensor,
        'metadata_tensor':  metadata_tensor,
        'beat2index':       beat2index,
        'index2beat':       index2beat,
        'pattern_counter':  dict(pattern_counter),
        'title':            title,
        'sequence_size':    sequence_size,
        'subdivision':      subdivision,  # 仅记录，与采样无关
    }


def main():
    parser = argparse.ArgumentParser(description='简谱整拍token预处理')
    parser.add_argument('--input', '-i', required=True, help='输入JSON文件路径')
    parser.add_argument('--output', '-o', default='./preprocessed_beat_token',
                        help='输出目录')
    parser.add_argument('--sequence_size', '-s', type=int, default=8,
                        help='每样本多少拍 (默认8)')
    parser.add_argument('--subdivision', '-d', type=int, default=8,
                        help='保留参数，写入元数据，不影响采样 (默认8)')
    parser.add_argument('--save_pt', action='store_true',
                        help='保存为 .pt 文件 (与 SimpleNotationDataset 兼容)')
    args = parser.parse_args()

    result = preprocess(
        json_path=args.input,
        sequence_size=args.sequence_size,
        subdivision=args.subdivision,
        verbose=True,
    )

    if args.save_pt:
        os.makedirs(args.output, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(args.output, f'{base}_beat_tensor_dataset.pt')
        torch.save({
            'chorale_tensor':   result['chorale_tensor'],
            'metadata_tensor':  result['metadata_tensor'],
            'note2index':       result['beat2index'],
            'index2note':       result['index2beat'],
            'sequence_size':    result['sequence_size'],
            # **关键**：beat-token 模式下 subdivision 恒为 1，"一拍 = 一个 token"。
            # SimpleNotationDataset.tensor_to_score 会调 beat_to_midi 把 pattern
            # 拆回真实的 sub-beat 音，所以输出 MIDI 时仍能保留 0.25 / 0.5 等细粒度。
            'subdivision':      1,
            'token_kind':       'beat',          # 标记这是 beat-token 数据
            'pattern_counter':  result['pattern_counter'],
            'title':            result['title'],
        }, out_path)
        print(f'已保存: {out_path}')
        print(f'  chorale_tensor : {result["chorale_tensor"].shape}')
        print(f'  metadata_tensor: {result["metadata_tensor"].shape}')
        print(f'  vocab size     : {len(result["beat2index"])}')


if __name__ == '__main__':
    main()
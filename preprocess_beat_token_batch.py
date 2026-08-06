"""
简谱"整拍 token"批量预处理
============================

把 `D:/code/music/sizhu_data/data/jin` 这类文件夹下的所有 JSON 简谱合并为一份
全局词表的训练集。流程与 `preprocess_beat_token.py` (单文件版) 完全一致，
唯一区别：每个文件的拍序列都映射到**同一个** vocab，而不是各自建一份。

输出 .pt 文件结构 (与单文件版相同 + 多文件标记):
    - chorale_tensor : (N, 1, sequence_size)         每元素 = beat-pattern 索引
    - metadata_tensor: (N, 1, sequence_size, 6)       [IsPlaying, Tick, Mode, Key, Fermata, voice_id]
    - note2index     : dict[pattern_str -> idx]
    - index2note     : dict[idx -> pattern_str]
    - sequence_size  : 每样本的拍数
    - subdivision    : 恒为 1 (一拍一个 token)
    - token_kind     : 'beat'
    - pattern_counter: dict[pattern -> 总出现次数]
    - source_files   : list[str]   参与训练的所有 JSON 路径

用法:
    python preprocess_beat_token_batch.py --input <dir> --output <out_dir>
    python preprocess_beat_token_batch.py --input D:/code/music/sizhu_data/data/jin
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

# 复用单文件版的切拍与编码逻辑
from preprocess_beat_token import notes_to_beats, beat_to_pattern


def process_one(json_path):
    """
    处理单个 JSON，返回:
        patterns : list[str]        一拍一个 pattern 字符串
        is_playing: list[int]      该拍是否含非休止音符
        n_beats  : int              拍数
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    beats = notes_to_beats(data.get('notes', []))
    patterns = [beat_to_pattern(b) for b in beats]
    is_playing = [
        1 if any(str(n.get('value')) != '0' for n in b) else 0
        for b in beats
    ]
    return patterns, is_playing, len(beats)


def main():
    parser = argparse.ArgumentParser(description='简谱整拍token批量预处理')
    parser.add_argument('--input', '-i', required=True,
                        help='输入 JSON 文件夹路径')
    parser.add_argument('--output', '-o', default='./preprocessed_beat_token',
                        help='输出目录')
    parser.add_argument('--sequence_size', '-s', type=int, default=8,
                        help='每样本多少拍 (默认8)')
    parser.add_argument('--suffix', default=None,
                        help='输出文件后缀 (默认取文件夹名)')
    parser.add_argument('--stride', type=int, default=1,
                        help='滑窗步长 (拍)，默认 1')
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.input, '*.json')))
    if not json_files:
        raise SystemExit(f'未发现 JSON: {args.input}')

    print('=' * 60)
    print(f'[整拍token批量预处理] {args.input}')
    print('=' * 60)
    print(f'发现 {len(json_files)} 个 JSON:')

    per_file_patterns = []
    per_file_is_playing = []
    total_pattern_counter = Counter()

    for jp in json_files:
        patterns, is_playing, n_beats = process_one(jp)
        per_file_patterns.append(patterns)
        per_file_is_playing.append(is_playing)
        total_pattern_counter.update(patterns)
        uniq = len(set(patterns))
        print(f'  {os.path.basename(jp):32s}  {n_beats:5d} 拍  '
              f'{uniq:3d} 唯一 pattern')

    # ---------- 全局词表 ----------
    sorted_patterns = sorted(total_pattern_counter.items(),
                             key=lambda x: (-x[1], x[0]))
    index2beat = {0: 'START', 1: 'END'}                # 与 DatasetManager.helpers 同名
    for pat, _ in sorted_patterns:
        if pat in ('START', 'END'):
            continue
        index2beat[len(index2beat)] = pat
    beat2index = {pat: i for i, pat in index2beat.items()}

    print()
    print(f'全局 vocab 大小: {len(beat2index)} (含 START/END)')
    print(f'Top-10 高频 pattern:')
    for pat, cnt in total_pattern_counter.most_common(10):
        idx = beat2index[pat]
        print(f'  [{idx:3d}]  {cnt:4d} ×  {pat}')

    # ---------- 滑动窗口采样 ----------
    MODE_MAJOR = 1
    KEY_D_MAJOR = 10
    chorale_list, meta_list = [], []

    file_sample_counts = []
    for patterns, is_playing in zip(per_file_patterns, per_file_is_playing):
        if len(patterns) < args.sequence_size:
            file_sample_counts.append(0)
            continue
        beat_ids = np.array([beat2index[p] for p in patterns], dtype=np.int64)
        n = len(beat_ids)
        is_play_arr = np.array(is_playing, dtype=np.int64)
        tick_arr = np.zeros(n, dtype=np.int64)
        mode_arr = np.full(n, MODE_MAJOR, dtype=np.int64)
        key_arr = np.full(n, KEY_D_MAJOR, dtype=np.int64)
        ferm_arr = np.zeros(n, dtype=np.int64)
        voice_arr = np.zeros(n, dtype=np.int64)

        cnt = 0
        for start in range(0, n - args.sequence_size + 1, args.stride):
            end = start + args.sequence_size
            win_ids = beat_ids[start:end]
            win_meta = np.stack([
                is_play_arr[start:end],
                tick_arr[start:end],
                mode_arr[start:end],
                key_arr[start:end],
                ferm_arr[start:end],
                voice_arr[start:end],
            ], axis=-1)
            chorale_list.append(win_ids[np.newaxis, :])
            meta_list.append(win_meta[np.newaxis, :, :])
            cnt += 1
        file_sample_counts.append(cnt)

    chorale_tensor = np.stack(chorale_list)
    metadata_tensor = np.stack(meta_list)
    print()
    print('每文件产出样本数:')
    for jp, cnt in zip(json_files, file_sample_counts):
        print(f'  {os.path.basename(jp):32s}  {cnt:5d} 样本')
    print(f'\n总样本数: {chorale_tensor.shape[0]}')

    # ---------- 保存 ----------
    suffix = args.suffix or os.path.basename(os.path.normpath(args.input))
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(
        args.output,
        f'dataset_{suffix}_combined_beat_tensor_dataset.pt',
    )
    payload = {
        'chorale_tensor':   chorale_tensor,
        'metadata_tensor':  metadata_tensor,
        'note2index':       beat2index,
        'index2note':       index2beat,
        'sequence_size':    args.sequence_size,
        'subdivision':      1,
        'token_kind':       'beat',
        'pattern_counter':  dict(total_pattern_counter),
        'source_files':     json_files,
    }
    torch.save(payload, out_path)
    print()
    print(f'已保存: {out_path}')
    print(f'  chorale_tensor : {chorale_tensor.shape}')
    print(f'  metadata_tensor: {metadata_tensor.shape}')
    print(f'  vocab size     : {len(beat2index)}')
    print(f'  文件数         : {len(json_files)}')


if __name__ == '__main__':
    main()
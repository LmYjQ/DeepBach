"""
Beat 模式约束生成
==================

这是 `generate_constrained.py` 的 beat-token 适配版。

约束语义 (与 tick 级不同)：
    - 每个约束点 = 一拍的 `ban=1` 或 `yan=1`
    - 模型只需保证"该 beat 内**第一个 sub-note 的 MIDI pitch** 与输入一致"
    - 其余 sub-note 的音高 / 时值自由发挥
    - 对应到 Gibbs 采样：约束位置直接预填，Gibbs 只在非约束位置更新

工作流:
    JSON (含 ban/yan 标记)
      ↓ notes_to_beats → 拍序列
      ↓ 找 ban/yan 在拍内的 "第一个 sub-note pitch"
      ↓ 从 vocab 里挑 "第一个 sub-note pitch 等于约束值" 的最高频 pattern → 预填
      ↓ deepbach.generation(tensor_chorale=预填, time_index_list_ticks=约束beats)
      ↓ tensor_to_score → music21 Score → MIDI

用法:
    python generate_constrained_beat.py \\
        --json  path/to/score.json \\
        --data  preprocessed/dataset_xxx_combined_beat_tensor_dataset.pt \\
        --model models/voicemodel_xxx_ne..._li..._0.pt \\
        --output generated.mid
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import music21

from preprocess_beat_token import notes_to_beats, note_to_pitch
from train_simple_notation import SimpleNotationDataset
from DeepBach.model_manager import DeepBach
from DeepBach.voice_model import parse_model_filename
from DatasetManager.helpers import beat_to_midi


# ---------- 约束点识别 ----------

def identify_constrained_beats(notes, mode='every_beat'):
    """
    从 JSON 音符列表中找出约束点。

    Args:
        mode:
          - 'every_beat' (默认): 每拍都按第一个 sub-note 的 pitch 约束
            — 因为 beat 模式下每拍天然是 "整数拍起点"，没有比这更细致的"on-beat"概念
          - 'ban_yan': 只把第一个 sub-note 的 ban=1 或 yan=1 当作约束点

    Returns:
        list of (beat_idx, first_midi_pitch, reason_str)
    """
    beats = notes_to_beats(notes)
    constraints = []
    for b_idx, beat in enumerate(beats):
        if not beat:
            continue
        first = beat[0]
        v_raw = first.get('value')
        if v_raw is None or str(v_raw).lower() in ('bar', 'space'):
            continue

        if mode == 'ban_yan':
            ban = int(first.get('ban', 0))
            yan = int(first.get('yan', 0))
            if not (ban == 1 or yan == 1):
                continue
            reason = 'ban' if ban == 1 else 'yan'
        else:  # 'every_beat'
            reason = 'beat_anchor'

        if str(v_raw) == '0':
            pitch = 0  # 休止符
        else:
            pitch = note_to_pitch(int(v_raw), first.get('octave', 0))
        constraints.append((b_idx, pitch, reason))
    return constraints


# ---------- 预填 ----------

def build_pitch_index(pattern_counter):
    """
    为快速查找，按 "每个 pattern 的第一个 sub-note pitch" 反向索引。
    Returns:
        dict[first_pitch -> list[(pattern_str, count)]]
    """
    pitch_to = defaultdict(list)
    for pat, cnt in pattern_counter.items():
        sub = beat_to_midi(pat)
        if not sub:
            continue
        first_pitch = sub[0][0]  # None for rest (we use 0 to mean rest in constraints)
        # beat_to_midi 返回 None 表休止符，但 pattern_counter 的 key 可能是 R@...
        # 我们用 "是否含 None" 区分：None=休止，数字=音高
        if first_pitch is None:
            key = 0  # 用 0 作为休止符 key
        else:
            key = int(first_pitch)
        pitch_to[key].append((pat, cnt))
    return pitch_to


def prefill_constrained_beats(beat2index, pattern_counter, constraints):
    """
    对每个约束点选一个 pattern 预填。

    选取策略：
        - 在 vocab 里挑 "第一个 sub-note pitch = 约束 pitch" 的所有 pattern
        - 取频次最高者作为预填值
        - 没有匹配时退化为 vocab 中频次最高的 pattern
    """
    pitch_index = build_pitch_index(pattern_counter)

    # 兜底：最高频 pattern
    fallback = max(pattern_counter.items(), key=lambda x: x[1])[0] if pattern_counter else None

    prefill = []
    for b_idx, pitch, reason in constraints:
        candidates = pitch_index.get(pitch, [])
        if candidates:
            best = max(candidates, key=lambda x: x[1])[0]
        else:
            best = fallback if fallback is not None else list(beat2index.keys())[0]
        prefill.append((b_idx, beat2index[best], reason, pitch))
    return prefill


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser(description='整拍token约束生成')
    parser.add_argument('--json', '-j', required=True, help='输入 JSON (含 ban/yan)')
    parser.add_argument('--data', '-d', required=True, help='训练数据 .pt 路径')
    parser.add_argument('--model', '-m', required=True, help='模型 .pt 路径')
    parser.add_argument('--output', '-o', default='generated_beat_constrained.mid',
                        help='输出 MIDI 路径')
    parser.add_argument('--subdivision', '-s', type=int, default=1,
                        help='与训练时一致 (beat 模式恒为 1)')
    parser.add_argument('--num_iterations', '-n', type=int, default=500,
                        help='Gibbs 迭代次数')
    parser.add_argument('--temperature', '-t', type=float, default=1.0)
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--sequence_length_ticks', type=int, default=None,
                        help='生成序列长度 (beats)，默认取 JSON 拍数向上取整到 sequence_size 倍数')
    parser.add_argument('--constraint_mode', default='every_beat',
                        choices=['every_beat', 'ban_yan'],
                        help='约束点选取方式 (默认 every_beat: 每拍首音都约束；ban_yan: 仅 ban/yan)')
    args = parser.parse_args()

    # ---- 1. 读 JSON 与训练数据 ----
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    notes = data.get('notes', [])
    beats = notes_to_beats(notes)
    total_beats = len(beats)
    print('=' * 60)
    print(f'[Beat 模式约束生成] {data.get("title", args.json)}')
    print('=' * 60)
    print(f'JSON 拍数: {total_beats}')

    dataset = SimpleNotationDataset(args.data, subdivision=args.subdivision)
    assert dataset.token_kind == 'beat', \
        f'此脚本要求 token_kind="beat"，当前 .pt 是 {dataset.token_kind!r}'
    beat2index = dataset.note2index_dicts[0]
    index2beat = dataset.index2note_dicts[0]
    pattern_counter = dataset.data.get('pattern_counter', {})

    # ---- 2. 模型 ----
    model_params = parse_model_filename(os.path.basename(args.model))
    assert model_params is not None, \
        f'无法从 {args.model} 文件名解析模型参数，请检查命名格式'
    print(f'从模型文件名解析参数: {model_params}')

    deepbach = DeepBach(
        dataset=dataset,
        note_embedding_dim=model_params['note_embedding_dim'],
        meta_embedding_dim=model_params['meta_embedding_dim'],
        num_layers=model_params['num_layers'],
        lstm_hidden_size=model_params['lstm_hidden_size'],
        dropout_lstm=model_params['dropout_lstm'],
        linear_hidden_size=model_params['hidden_size_linear'],
        model_suffix='',   # 直接用 model_path
        models_dir='',
    )
    deepbach.load(model_path=args.model)
    deepbach.cuda()

    # ---- 3. 序列长度 ----
    seq_len = args.sequence_length_ticks
    if seq_len is None:
        # JSON 拍数向上对齐到 sequence_size 倍数
        seq_len = ((total_beats + dataset.sequence_size - 1)
                   // dataset.sequence_size) * dataset.sequence_size
    if seq_len < total_beats:
        seq_len = ((total_beats + dataset.sequence_size - 1)
                   // dataset.sequence_size) * dataset.sequence_size
    print(f'生成序列长度: {seq_len} beats')

    # ---- 4. 识别约束点 ----
    constraints = identify_constrained_beats(notes, mode=args.constraint_mode)
    print(f'约束模式: {args.constraint_mode}')
    # 只考虑生成范围内的约束
    constraints_in_range = [(b, p, r) for b, p, r in constraints if b < seq_len]
    print(f'约束点数: {len(constraints_in_range)}')
    for b, p, r in constraints_in_range[:5]:
        sub = beats[b]
        first = sub[0]
        v = first.get('value')
        o = first.get('octave', 0)
        print(f'  beat {b:3d}  ({r})  first_note=value={v} octave={o}  '
              f'midi={p}  内含 {len(sub)} 个 sub-note')
    if len(constraints_in_range) > 5:
        print(f'  ... 共 {len(constraints_in_range)} 个')

    # ---- 5. 预填 chorale tensor ----
    prefill = prefill_constrained_beats(beat2index, pattern_counter, constraints_in_range)
    init_chorale = dataset.random_score_tensor(seq_len)  # (1, seq_len)
    print()
    print('预填结果 (约束 beat → vocab pattern):')
    for b_idx, token_id, reason, pitch in prefill:
        pat = index2beat[token_id]
        init_chorale[0, b_idx] = token_id
        print(f'  beat {b_idx:3d}  pitch={pitch:3d}  token_id={token_id:3d}  '
              f'pat="{pat}"')

    # ---- 6. metadata tensor ----
    # 训练时 SimpleNotationDataset 把 6 列 metadata 切成 [Tick, Mode, Key, Fermata, voice_id]
    # (剔除 IsPlaying)。VoiceModel 也按 5 个 meta-embedding 建模，所以这里必须传 5 列。
    seq_metadata = torch.zeros(1, seq_len, 5, dtype=torch.long)
    seq_metadata[0, :, 0] = 0              # Tick (一拍一个 token, 填 0)
    seq_metadata[0, :, 1] = 1              # Mode = major
    seq_metadata[0, :, 2] = 10             # Key = D major
    seq_metadata[0, :, 3] = 0              # Fermata = 0
    seq_metadata[0, :, 4] = 0              # voice_id

    # ---- 7. 调用 generation ----
    constrained_beats = [b for b, _, _ in constraints_in_range]
    print()
    print(f'开始 Gibbs 采样: num_iterations={args.num_iterations}, '
          f'batch_size_per_voice={args.batch_size}, '
          f'constrained_beats={len(constrained_beats)}')

    score, final_chorale, _ = deepbach.generation(
        tensor_chorale=init_chorale,           # (1, seq_len)
        tensor_metadata=seq_metadata,           # (1, seq_len, 6)
        temperature=args.temperature,
        batch_size_per_voice=args.batch_size,
        num_iterations=args.num_iterations,
        sequence_length_ticks=seq_len,
        time_index_range_ticks=[0, seq_len],    # 整段都可采样 (约束点会被跳过)
        time_index_list_ticks=constrained_beats,
        random_init=False,                     # 已经手动初始化，不要再随机化
    )

    # ---- 8. 验证约束 ----
    print()
    print('约束保持验证 (检查最终 tensor 中约束 beat 的第一个 sub-note pitch):')
    all_ok = True
    fail = 0
    for b_idx, expected_token_id, reason, pitch in prefill:
        actual_token_id = final_chorale[0, b_idx].item()
        actual_pat = index2beat[actual_token_id]
        sub = beat_to_midi(actual_pat)
        actual_first = sub[0][0] if sub else None
        actual_first_int = 0 if actual_first is None else int(actual_first)
        ok = (actual_first_int == pitch)
        if not ok:
            mark = '✗'
            fail += 1
            if fail <= 5:
                print(f'  beat {b_idx:3d} [{mark}]  expected_pitch={pitch:3d}  '
                      f'actual_pitch={actual_first_int:3d}  pat="{actual_pat}"')
        else:
            mark = '✓'
        if fail == 0 and b_idx == prefill[-1][0]:
            print(f'  beat {b_idx:3d} [{mark}]  expected_pitch={pitch:3d}  '
                  f'actual_pitch={actual_first_int:3d}  pat="{actual_pat}"')
        if not ok:
            all_ok = False
    print(f'\n约束统计: 全部 {len(prefill)} 个约束点中失败 {fail} 个')
    print(f'全部约束保持: {all_ok}')

    # ---- 9. 保存 ----
    score.write('midi', fp=args.output)
    print()
    print(f'保存: {args.output}')


if __name__ == '__main__':
    main()
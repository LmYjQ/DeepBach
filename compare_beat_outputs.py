"""
对比 4 个模型在 zhouBY_slice.json 上的 ban/yan 约束生成结果。
统计: pitch range / 唯一 pattern 数 / 高频 pattern 差异 / 与 zhou 训练的 pattern 相似度
"""
import os, json
from collections import Counter

import torch
from music21 import converter

from preprocess_beat_token import notes_to_beats, beat_to_pattern, note_to_pitch
from train_simple_notation import SimpleNotationDataset
from DatasetManager.helpers import beat_to_midi


def analyze_midi(midi_path, dataset_pt_path, label):
    score = converter.parse(midi_path)
    events = list(score.flatten().notesAndRests)

    # 按 1.0 拍重组
    beats_reconstructed = []
    cur = []
    for e in events:
        cur.append(e)
        if abs(sum(x.duration.quarterLength for x in cur) - 1.0) < 1e-3:
            beats_reconstructed.append(cur)
            cur = []
    if cur:
        beats_reconstructed.append(cur)

    # pattern 统计
    patterns = []
    for beat in beats_reconstructed:
        parts = []
        for e in beat:
            if e.isNote:
                parts.append(f"{int(e.pitch.ps)}@{e.duration.quarterLength:g}")
            else:
                parts.append(f"R@{e.duration.quarterLength:g}")
        patterns.append("|".join(parts))
    counter = Counter(patterns)

    # pitch range
    pitches = [int(e.pitch.ps) for e in events if e.isNote]
    return {
        'label':        label,
        'events':       len(events),
        'beats':        len(beats_reconstructed),
        'pitches':      pitches,
        'pitch_min':    min(pitches) if pitches else 0,
        'pitch_max':    max(pitches) if pitches else 0,
        'unique_pat':   len(counter),
        'top5_pat':     counter.most_common(5),
        'pat_counter':  counter,
    }


def analyze_dataset_pt(dataset_pt_path, label):
    ds = SimpleNotationDataset(dataset_pt_path)
    # 把 chorale_tensor 重新解码成 patterns (取前 200 拍做参考)
    pat_counter = Counter()
    sample = torch.from_numpy(ds.chorale_tensor[:, 0, :]).long()
    for s_idx in range(min(sample.shape[0], 100)):
        idx2beat = ds.index2note_dicts[0]
        for i in range(sample.shape[1]):
            token_id = sample[s_idx, i].item()
            pat = idx2beat[token_id]
            if pat not in ('START', 'END'):
                pat_counter[pat] += 1
    return {
        'label':       label,
        'pat_counter': pat_counter,
        'vocab_size':  len(ds.note2index_dicts[0]),
        'num_samples': ds.chorale_tensor.shape[0],
    }


# ---- 原始 JSON 的 ground truth patterns ----
with open('D:/code/music/sizhu_data/data/generate_input/zhouBY_slice.json', encoding='utf-8') as f:
    zhou_orig = json.load(f)
notes_orig = [n for n in zhou_orig.get('notes', []) if n.get('value') not in ('bar', None)]
orig_beats = notes_to_beats(zhou_orig.get('notes', []))
orig_patterns = [beat_to_pattern(b) for b in orig_beats]
orig_pitches = []
for b in orig_beats:
    if b:
        first = b[0]
        v = str(first.get('value'))
        if v not in ('bar', 'space', '0'):
            orig_pitches.append(note_to_pitch(int(v), first.get('octave', 0)))

print('=' * 70)
print('原始 JSON (zhouBY_slice) 参考')
print('=' * 70)
print(f'  拍数: {len(orig_patterns)}')
print(f'  唯一 pattern: {len(set(orig_patterns))}')
print(f'  首音 range: {min(orig_pitches)} ~ {max(orig_pitches)}')
print(f'  Top-5 pattern:')
for p, c in Counter(orig_patterns).most_common(5):
    print(f'    {c:3d} × {p}')


# ---- 4 个数据集的训练 vocab ----
print()
print('=' * 70)
print('4 个训练数据集的 vocab 概况')
print('=' * 70)
datasets = {}
for ds_name in ['jin', 'tang', 'zhou', 'lu']:
    info = analyze_dataset_pt(
        f'./preprocessed_beat_token/dataset_{ds_name}_combined_beat_tensor_dataset.pt',
        ds_name)
    datasets[ds_name] = info
    print(f'  {ds_name:6s}: vocab={info["vocab_size"]:3d}, samples={info["num_samples"]}')


# ---- 4 个生成结果 ----
print()
print('=' * 70)
print('4 个模型在 zhouBY_slice.json 上的生成结果 (ban_yan 约束)')
print('=' * 70)

results = {}
gen_specs = [
    ('jin',  './generated_jin.mid',  './preprocessed_beat_token/dataset_jin_combined_beat_tensor_dataset.pt'),
    ('tang', './generated_tang.mid', './preprocessed_beat_token/dataset_tang_combined_beat_tensor_dataset.pt'),
    ('zhou', './generated_zhou.mid', './preprocessed_beat_token/dataset_zhou_combined_beat_tensor_dataset.pt'),
    ('lu',   './generated_lu.mid',   './preprocessed_beat_token/dataset_lu_combined_beat_tensor_dataset.pt'),
]
for label, midi, pt in gen_specs:
    info = analyze_midi(midi, pt, label)
    results[label] = info
    print(f'\n[{label}] 训练数据 {datasets[label]["vocab_size"]} vocab')
    print(f'  MIDI 大小    : {os.path.getsize(midi)} bytes')
    print(f'  beats 重组成: {info["beats"]}')
    print(f'  events 数   : {info["events"]}')
    print(f'  pitch range : {info["pitch_min"]} ~ {info["pitch_max"]}')
    print(f'  唯一 pattern: {info["unique_pat"]}')
    print(f'  Top-5 pattern:')
    for p, c in info['top5_pat']:
        print(f'    {c:3d} × {p}')


# ---- 与训练 vocab 的重叠度 ----
print()
print('=' * 70)
print('生成 pattern vs 训练 vocab 重叠度 (看模型是否在用训练集见过的 pattern)')
print('=' * 70)
for label, info in results.items():
    gen_set = set(info['pat_counter'].keys())
    train_set = set(datasets[label]['pat_counter'].keys())
    overlap = gen_set & train_set
    novel = gen_set - train_set
    coverage = len(overlap) / max(1, len(gen_set))
    print(f'  [{label}] 用了 {len(gen_set)} 个 pattern, '
          f'其中 {len(overlap)} ({coverage:.0%}) 在训练 vocab 里, '
          f'{len(novel)} 个是 vocab 外 ({len(novel)/max(1,len(gen_set)):.0%})')


# ---- 与原始 zhouBY_slice 模式的相似度 ----
print()
print('=' * 70)
print('生成 pattern vs 原始 JSON pattern 重叠度')
print('=' * 70)
orig_set = set(orig_patterns)
for label, info in results.items():
    gen_set = set(info['pat_counter'].keys())
    overlap = gen_set & orig_set
    coverage = len(overlap) / max(1, len(orig_set))
    print(f'  [{label}] 生成模式与原曲有 {len(overlap)}/{len(orig_set)} ({coverage:.0%}) 重叠')
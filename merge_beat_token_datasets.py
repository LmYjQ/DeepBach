"""
合并多个 beat-token .pt 文件到一份全局 vocab 的 .pt。

设计:
    - 全局 vocab = 各文件 vocab 的并集 (START/END 已对齐)
    - 各文件的 token id 重新映射到全局 id
    - chorale_tensor / metadata_tensor 拼接为一份
    - pattern_counter 累加
    - source_files 列表拼接

用法:
    python merge_beat_token_datasets.py \\
        --inputs  preprocessed_beat_token/dataset_jin_..._beat_tensor_dataset.pt \\
                  preprocessed_beat_token/dataset_tang_..._beat_tensor_dataset.pt \\
                  preprocessed_beat_token/dataset_zhou_..._beat_tensor_dataset.pt \\
        --output  preprocessed_beat_token/dataset_all_combined_beat_tensor_dataset.pt
"""

import argparse
import os
from collections import Counter

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description='合并多个 beat-token .pt')
    parser.add_argument('--inputs', '-i', nargs='+', required=True,
                        help='输入 .pt 路径列表')
    parser.add_argument('--output', '-o', required=True, help='输出 .pt 路径')
    parser.add_argument('--name', default='all',
                        help='输出文件后缀标识 (默认 all)')
    args = parser.parse_args()

    print('=' * 60)
    print(f'合并 {len(args.inputs)} 个 beat-token .pt')
    print('=' * 60)

    payloads = []
    for p in args.inputs:
        d = torch.load(p, map_location='cpu', weights_only=False)
        print(f'  {os.path.basename(p)}')
        print(f'    chorale: {d["chorale_tensor"].shape}, vocab={len(d["note2index"])}')
        payloads.append(d)
        assert d.get('token_kind') == 'beat', f'{p} 不是 beat-token 数据'

    # ---------- 全局 vocab ----------
    # START/END 永远在 index 0,1
    global_index2pat = {0: 'START', 1: 'END'}
    for d in payloads:
        for idx, pat in d['index2note'].items():
            if pat in ('START', 'END'):
                continue
            if pat not in global_index2pat.values():
                global_index2pat[len(global_index2pat)] = pat
    global_pat2idx = {p: i for i, p in global_index2pat.items()}
    print(f'\n全局 vocab 大小: {len(global_pat2idx)} (含 START/END)')

    # ---------- 重新映射 token id 并拼接 ----------
    new_chorale = []
    new_meta = []
    new_pat_counter = Counter()
    all_sources = []

    for d in payloads:
        local2global = {}
        for local_idx, pat in d['index2note'].items():
            local2global[local_idx] = global_pat2idx[pat]

        local_chorale = d['chorale_tensor']  # (N, 1, L)
        # 重新映射: 用 numpy 索引快一些
        max_local = max(local2global.keys()) + 1
        remap = np.zeros(max_local, dtype=np.int64)
        for k, v in local2global.items():
            remap[k] = v
        remapped = remap[local_chorale]
        new_chorale.append(remapped)
        new_meta.append(d['metadata_tensor'])
        new_pat_counter.update(d.get('pattern_counter', {}))
        all_sources.extend(d.get('source_files', []))

    chorale_tensor = np.concatenate(new_chorale, axis=0)
    metadata_tensor = np.concatenate(new_meta, axis=0)
    print(f'合并后 chorale_tensor: {chorale_tensor.shape}')
    print(f'合并后 metadata_tensor: {metadata_tensor.shape}')
    print(f'来源文件总数: {len(all_sources)}')

    # ---------- 保存 ----------
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'chorale_tensor':   chorale_tensor,
        'metadata_tensor':  metadata_tensor,
        'note2index':       global_pat2idx,
        'index2note':       global_index2pat,
        'sequence_size':    payloads[0]['sequence_size'],
        'subdivision':      1,
        'token_kind':       'beat',
        'pattern_counter':  dict(new_pat_counter),
        'source_files':     all_sources,
    }, args.output)
    print(f'\n已保存: {args.output}')


if __name__ == '__main__':
    main()
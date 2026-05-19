"""
将简谱JSON格式转换为DeepBach模型所需的tensor格式

输入格式 (简谱JSON):
  - value: 1-7 (do=1, re=2, ..., si=7) 表示音高
  - octave: 整数，表示八度偏移 (0=中央八度)
  - duration: 时值 (1=一拍, 0.5=半拍等)
  - 其他字段暂不使用

输出格式 (DeepBach TensorDataset):
  - chorale_tensor: (N, 1, 32)  # N样本, 1声部, 32 ticks
  - metadata_tensor: (N, 1, 32, 2)  # tick + voice_id

用法:
  python preprocess_chinese_score.py
"""
import json
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm

# 简谱音名到 MIDI 的映射 (首调，中央C为基准)
# value 1-7 对应 do, re, mi, fa, sol, la, si
# octave=0 表示中央八度 (C4 = MIDI 60)
SOLFEGE_TO_MIDI_BASE = {
    1: 60,   # do -> C4 (中央C)
    2: 62,   # re -> D4
    3: 64,   # mi -> E4
    4: 65,   # fa -> F4
    5: 67,   # sol -> G4
    6: 69,   # la -> A4
    7: 71,   # si -> B4
}


def load_json_dataset(json_path):
    """加载简谱JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def json_to_midi_sequence(notes, subdivision=4):
    """
    将简谱JSON转换为MIDI音高序列

    Args:
        notes: JSON中的notes数组
        subdivision: 每拍的tick数 (默认4表示16分音符)

    Returns:
        midi_sequence: 音高序列 (以tick为单位，0表示休止符)
        duration_sequence: 时值序列
        is_articulated: 是否为音符起点的序列 (1=起点, 0=延续)
    """
    midi_sequence = []
    duration_sequence = []
    is_articulated = []

    current_tick = 0

    for note_data in notes:
        value = int(note_data['value'])
        octave = note_data['octave']
        duration = note_data['duration']

        # value=0 表示休止符
        if value == 0:
            duration_ticks = int(duration * subdivision)
            for tick in range(duration_ticks):
                midi_sequence.append(0)  # 0 表示休止符
                duration_sequence.append(duration_ticks - tick)
                is_articulated.append(1)  # 休止符算一个新起点
            current_tick += duration_ticks
            continue

        # 计算MIDI音高
        base_midi = SOLFEGE_TO_MIDI_BASE[value]
        midi_pitch = base_midi + (octave * 12)

        # 计算时值 (以tick为单位)
        duration_ticks = int(duration * subdivision)

        # 添加音符 (一个音符可能持续多个tick)
        for tick in range(duration_ticks):
            midi_sequence.append(midi_pitch)
            duration_sequence.append(duration_ticks - tick)
            is_articulated.append(1 if tick == 0 else 0)  # 第一个tick是起点，后续是延续

        # 移动到下一个位置
        current_tick += duration_ticks

    return np.array(midi_sequence), np.array(duration_sequence), np.array(is_articulated), current_tick


def create_tensor_dataset(midi_sequence, duration_sequence, is_articulated, sequence_size=8, subdivision=4, voice_id=0):
    """
    创建DeepBach格式的TensorDataset

    Args:
        midi_sequence: MIDI音高序列
        duration_sequence: 时值序列
        is_articulated: 是否为音符起点的序列 (1=起点, 0=延续)
        sequence_size: 每个样本的序列长度 (以拍为单位)
        subdivision: 每拍的tick数
        voice_id: 声部ID

    Returns:
        chorale_tensor: (N, 1, sequence_size * subdivision) 每一个元素是音高，或者slur延长记号， 长度是节拍数*细分数
        metadata_tensor: (N, 1, sequence_size * subdivision, 2) 每个元素是[tick_position, 声部编号]，tick_position按照细分数递增，表示时值到多少了
    """
    from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, REST_SYMBOL

    # 建立音符到索引的映射
    note_set = {SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, REST_SYMBOL}

    # 遍历所有音符，建立映射（跳过0=休止符）
    unique_pitches = set(midi_sequence) - {0}  # 0 表示休止符，不作为音高
    for pitch in unique_pitches:
        note_name = f"{pitch}"  # 用MIDI数字作为音符名
        note_set.add(note_name)

    # 创建映射表
    # note_list = sorted(list(note_set), key=lambda x: (
    #     0 if x == SLUR_SYMBOL else
    #     1 if x == START_SYMBOL else
    #     2 if x == END_SYMBOL else
    #     3 if x == REST_SYMBOL else
    #     4 if isinstance(x, str) else
    #     5
    # ))

    # 重新排列，把特殊符号放前面
    special_symbols = [SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, REST_SYMBOL]
    special_symbols.sort()

    # 构建映射
    index2note = {}
    note2index = {}

    idx = 0
    for sym in special_symbols:
        index2note[idx] = sym
        note2index[sym] = idx
        idx += 1

    for pitch in sorted(unique_pitches):
        note_name = str(pitch)
        index2note[idx] = note_name
        note2index[note_name] = idx
        idx += 1

    # 确保 REST_SYMBOL (0 休止符) 在映射表中
    if REST_SYMBOL not in note2index:
        index2note[idx] = REST_SYMBOL
        note2index[REST_SYMBOL] = idx
        idx += 1

    print(f"音符表大小: {len(note2index)}")
    print(f"音符表示例: {list(note2index.items())[:10]}")

    # 创建序列
    sequence_length_ticks = sequence_size * subdivision
    chorale_tensor_dataset = []
    metadata_tensor_dataset = []

    total_ticks = len(midi_sequence)

    print(f"\n创建训练样本...")
    print(f"  总长度: {total_ticks} ticks = {total_ticks / subdivision} 拍")
    print(f"  序列长度: {sequence_size} 拍 = {sequence_length_ticks} ticks")
    print(f"  subdivision: {subdivision} (每拍{subdivision}个tick)")

    # 滑动窗口采样
    # 步长为1拍
    for start_tick in range(0, total_ticks - sequence_length_ticks + 1, subdivision):
        end_tick = start_tick + sequence_length_ticks

        # 提取这个窗口的音符和是否起点
        window_pitches = midi_sequence[start_tick:end_tick]
        window_articulated = is_articulated[start_tick:end_tick]

        # 转换为索引，使用 SLUR_SYMBOL 表示音符延续
        voice_tensor = []
        for i in range(len(window_pitches)):
            pitch = window_pitches[i]
            articulated = window_articulated[i]

            if pitch == 0:
                # 休止符
                idx = note2index.get(REST_SYMBOL, note2index.get('0'))
            elif articulated == 0:
                # 延续音符
                idx = note2index.get(SLUR_SYMBOL)
            else:
                # 新音符起点
                note_name = str(pitch)
                idx = note2index.get(note_name, note2index.get(REST_SYMBOL))
            voice_tensor.append(idx)

        voice_tensor = np.array(voice_tensor).reshape(1, -1)  # (1, ticks)
        
        # 创建metadata (tick位置)
        tick_positions = np.arange(sequence_length_ticks) % subdivision
        voice_metadata = np.stack([
            tick_positions,
            np.full(sequence_length_ticks, voice_id)
        ], axis=1)

        chorale_tensor_dataset.append(voice_tensor)
        metadata_tensor_dataset.append(voice_metadata)
        # print(chorale_tensor_dataset)
        # print(metadata_tensor_dataset)
    # 合并
    chorale_tensor = np.array(chorale_tensor_dataset)
    metadata_tensor = np.array(metadata_tensor_dataset)

    print(f"\n输出形状:")
    print(f"  chorale_tensor: {chorale_tensor.shape}")
    print(f"  metadata_tensor: {metadata_tensor.shape}")

    return (chorale_tensor, metadata_tensor,
            note2index, index2note)


def preprocess_json_file(json_path, output_dir, sequence_size=8, subdivision=4):
    """
    预处理单个JSON文件

    Args:
        json_path: 输入JSON文件路径
        output_dir: 输出目录
        sequence_size: 序列长度 (拍)
        subdivision: 每拍tick数
    """
    print("=" * 60)
    print(f"处理文件: {json_path}")
    print("=" * 60)

    # 1. 加载JSON
    data = load_json_dataset(json_path)
    title = data.get('title', 'unknown')
    tempo = data.get('tempo', 60)
    beats_per_bar = data.get('beatsPerBar', 4)
    notes = data.get('notes', [])

    # 过滤掉小节线等非音符数据
    notes = [n for n in notes if str(n.get('value', '')).lower() != 'bar']
    notes = [n for n in notes if n.get('value') is not None]  # 过滤value为空的

    print(f"\n[1] 文件信息:")
    print(f"    标题: {title}")
    print(f"    节拍: {tempo} BPM")
    print(f"    每小节拍数: {beats_per_bar}")
    print(f"    音符数(过滤后): {len(notes)}")

    # 2. 转换为MIDI序列
    midi_seq, dur_seq, is_articulated, total_ticks = json_to_midi_sequence(notes, subdivision)

    print(f"\n[2] MIDI序列:")
    print(f"    总长度: {total_ticks} ticks = {total_ticks / subdivision} 拍")
    print(f"    前20个MIDI音高: {midi_seq[:20].tolist()}")
    print(f"    前20个时值: {dur_seq[:20].tolist()}")
    print(f"    前20个是否起点: {is_articulated[:20].tolist()}")
    

    # # 3. 打印前几个音符预览
    # print(f"\n[3] 前20个音符预览:")
    # for i in range(min(20, len(midi_seq))):
    #     dur = dur_seq[i]
    #     print(f"    tick {i}: MIDI {midi_seq[i]}, 剩余时值 {dur}")

    # 4. 创建TensorDataset
    chorale_tensor, metadata_tensor, note2index, index2note = create_tensor_dataset(
        midi_seq, dur_seq, is_articulated, sequence_size=sequence_size, subdivision=subdivision
    )

    # 5. 保存
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_tensor_dataset.pt")

    dataset_dict = {
        'title': title,
        'note2index': note2index,
        'index2note': index2note,
        'sequence_size': sequence_size,
        'subdivision': subdivision,
        'chorale_tensor': chorale_tensor,
        'metadata_tensor': metadata_tensor,
    }

    torch.save(dataset_dict, output_path)
    print(f"\n[5] 已保存: {output_path}")
    print(f"    样本数: {len(chorale_tensor)}")

    return dataset_dict


def midi_to_solfege(midi_pitch):
    """将MIDI音高还原为简谱格式 (value, octave)"""
    if midi_pitch == 0:
        return 0, 0  # 休止符

    # MIDI 60 = 中央C (do, octave=0)
    # 计算相对于中央C的半音数
    rel = midi_pitch - 60

    # 找到最近的自然音
    # C=0, D=2, E=4, F=5, G=7, A=9, B=11
    natural_notes = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
    note_idx = rel % 12
    octave_offset = rel // 12

    if note_idx in natural_notes:
        value = natural_notes[note_idx]
        octave = octave_offset
        return value, octave
    else:
        # 升降音处理：找到最近的自然音
        lower_idx = max([k for k in natural_notes if k <= note_idx])
        upper_idx = min([k for k in natural_notes if k >= note_idx])
        if note_idx - lower_idx <= upper_idx - note_idx:
            value = natural_notes[lower_idx]
        else:
            value = natural_notes[upper_idx]
        octave = octave_offset
        return value, octave


def verify_tensor_dataset(json_path, tensor_path, subdivision=8, n=20):
    """
    验证处理后的tensor是否正确
    - 合并相邻相同音
    - 还原简谱的01234567和duration, octave
    """
    print("\n" + "=" * 60)
    print("验证处理结果")
    print("=" * 60)

    # 加载原始JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    notes = data['notes']

    # 过滤非音符数据
    original_notes = [n for n in notes if str(n.get('value', '')).lower() != 'bar' and n.get('value') is not None]

    # 加载处理后的tensor
    dataset = torch.load(tensor_path, weights_only=False)

    print(f"\n原始JSON音符数: {len(original_notes)}")

    chorale_tensor = dataset['chorale_tensor']
    metadata_tensor = dataset['metadata_tensor']
    index2note = dataset['index2note']

    from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, REST_SYMBOL

    print(f"chorale_tensor (前{n}行):")
    for i, row in enumerate(chorale_tensor[:n]):
        row_data = row[0].tolist()  # (64,) 音高索引序列
        # 转为简谱格式 {value}_{octave}
        original_list = []
        for idx in row_data:
            note_name = index2note.get(int(idx), '?')
            if note_name == REST_SYMBOL:
                original_list.append('0_0')  # 休止符
            elif note_name == SLUR_SYMBOL:
                original_list.append('_')  # 延长记号
            elif note_name in [START_SYMBOL, END_SYMBOL]:
                original_list.append(note_name)
            else:
                pitch = int(note_name) if note_name.isdigit() else 0
                if pitch > 0:
                    value, octave = midi_to_solfege(pitch)
                    original_list.append(f'{value}-{octave}')
                else:
                    original_list.append('0_0')

        # metadata
        meta_row = metadata_tensor[i].tolist()  # (64, 2)
        tick_list = [str(x[0]) for x in meta_row]
        voice_list = [str(x[1]) for x in meta_row]

        print(f"样本{i}:")
        print(f"  chorale:     {','.join(str(x) for x in row_data)}")
        print(f"  original:    {','.join(original_list)}")
        print(f"  tick:        {','.join(tick_list)}")
        print(f"  voice:       {','.join(voice_list)}")

    # 提取序列逻辑已注释
    # # 反向映射: index -> pitch
    # from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, REST_SYMBOL
    #
    # # 合并相邻相同音
    # print("\n" + "-" * 40)
    # print("合并相邻tick的相同音:")
    # print("-" * 40)
    #
    # # 从第一个样本提取完整序列
    # sample_0 = chorale_tensor[0, 0, :]
    # if isinstance(sample_0, torch.Tensor):
    #     sample_0 = sample_0.numpy()
    # total_ticks = len(sample_0)
    #
    # merged_notes = []
    # prev_pitch, current_pitch = None, None
    # current_duration = 0
    #
    # for tick in range(total_ticks):
    #     idx = sample_0[tick]
    #     note_name = index2note.get(int(idx), '?')
    #     if note_name == SLUR_SYMBOL:
    #         # 延续音符，增加duration
    #         current_duration += 1
    #     elif note_name in [START_SYMBOL, END_SYMBOL]:
    #         # 忽略
    #         continue
    #     elif note_name == REST_SYMBOL:
    #         # 休止符：结束当前音，开始休止
    #         if current_pitch is not None:
    #             value, octave = midi_to_solfege(current_pitch)
    #             merged_notes.append({
    #                 'tick': tick - current_duration,
    #                 'value': value,
    #                 'octave': octave,
    #                 'duration': current_duration / subdivision,
    #                 'midi': current_pitch
    #             })
    #             current_pitch = None
    #             current_duration = 0
    #         current_duration += 1
    #     else:
    #         # 实际音高
    #         pitch = int(note_name) if note_name.isdigit() else 0

    print(f"处理后样本数: {chorale_tensor.shape[0]}")
    print(f"每个样本: {chorale_tensor.shape[1]} 声部, {chorale_tensor.shape[2]} ticks")

    # 反向映射: index -> pitch
    # from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, REST_SYMBOL

    # # 合并相邻相同音
    # print("\n" + "-" * 40)
    # print("合并相邻tick的相同音:")
    # print("-" * 40)

    # # 从第一个样本提取完整序列
    # sample_0 = chorale_tensor[0, 0, :]
    # if isinstance(sample_0, torch.Tensor):
    #     sample_0 = sample_0.numpy()
    # total_ticks = len(sample_0)

    # merged_notes = []
    # prev_pitch, current_pitch = None, None
    # current_duration = 0

    # for tick in range(total_ticks):
    #     idx = sample_0[tick]
    #     note_name = index2note.get(int(idx), '?')
    #     if note_name == SLUR_SYMBOL:
    #         # 延续音符，增加duration
    #         current_duration += 1
    #     elif note_name in [START_SYMBOL, END_SYMBOL]:
    #         # 忽略
    #         continue
    #     elif note_name == REST_SYMBOL:
    #         # 休止符：结束当前音，开始休止
    #         if current_pitch is not None:
    #             value, octave = midi_to_solfege(current_pitch)
    #             merged_notes.append({
    #                 'tick': tick - current_duration,
    #                 'value': value,
    #                 'octave': octave,
    #                 'duration': current_duration / subdivision,
    #                 'midi': current_pitch
    #             })
    #             current_pitch = None
    #             current_duration = 0
    #         current_duration += 1
    #     else:
    #         # 实际音高
    #         pitch = int(note_name) if note_name.isdigit() else 0

    #         if pitch == current_pitch:
    #             # 相同音持续（SLUR处理）
    #             current_duration += 1
    #         else:
    #             # 新音符：先结束上一个
    #             if current_pitch is not None:
    #                 value, octave = midi_to_solfege(current_pitch)
    #                 merged_notes.append({
    #                     'tick': tick - current_duration,
    #                     'value': value,
    #                     'octave': octave,
    #                     'duration': current_duration / subdivision,
    #                     'midi': current_pitch
    #                 })
    #             # 开始新音符
    #             current_pitch = pitch
    #             current_duration = 1

    # # 处理最后一个音符
    # if current_pitch is not None:
    #     value, octave = midi_to_solfege(current_pitch)
    #     merged_notes.append({
    #         'tick': total_ticks - current_duration,
    #         'value': value,
    #         'octave': octave,
    #         'duration': current_duration / subdivision,
    #         'midi': current_pitch
    #     })

    # print(f"\n合并后音符数: {len(merged_notes)}")
    # print("\n前20个音符 (还原简谱格式):")
    # print(f"{'tick':>6} {'value':>6} {'octave':>7} {'duration':>8} {'MIDI':>6}")
    # print("-" * 40)
    # for i, note in enumerate(merged_notes[:20]):
    #     print(f"{note['tick']:>6} {note['value']:>6} {note['octave']:>7} {note['duration']:>8.2f} {note['midi']:>6}")

    # # 与原始JSON对比
    # print("\n" + "-" * 40)
    # print("与原始JSON对比 (前10个音符):")
    # print("-" * 40)
    # print(f"{'原始value':>10} {'原始octave':>11} {'原始dur':>9} | {'还原value':>10} {'还原octave':>11} {'还原dur':>9}")
    # print("-" * 60)
    # for i, orig in enumerate(original_notes[:10]):
    #     if i < len(merged_notes):
    #         note = merged_notes[i]
    #         orig_value = int(orig.get('value')) if orig.get('value') is not None else None
    #         orig_octave = int(orig.get('octave', 0)) if orig.get('octave') is not None else 0
    #         match = "OK" if (orig_value == note['value'] and orig_octave == note['octave']) else "FAIL"
    #         print(f"{orig_value:>10} {orig_octave:>11} {orig.get('duration'):>9.2f} | "
    #               f"{note['value']:>10} {note['octave']:>11} {note['duration']:>9.2f} {match}")
    #     else:
    #         print(f"{int(orig.get('value')):>10} {int(orig.get('octave', 0)):>11} {orig.get('duration'):>9.2f} | (无对应)")

    return dataset


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='预处理简谱JSON为DeepBach格式')
    parser.add_argument('--input', '-i', default=None,
                        help='输入JSON文件路径')
    parser.add_argument('--output', '-o', default='./preprocessed_data',
                        help='输出目录')
    parser.add_argument('--sequence_size', '-s', type=int, default=8,
                        help='序列长度(拍), 默认8')
    parser.add_argument('--subdivision', '-d', type=int, default=8,
                        help='每拍tick数, 默认8 (32分音符)')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='跳过预处理步骤')
    parser.add_argument('--no-verify', action='store_true',
                        help='跳过验证步骤')

    args = parser.parse_args()

    # 如果没有指定输入，使用默认的JSON文件
    if args.input is None:
        default_paths = [
            r'D:\code\music\qmx_reader\dataset_da\云庆5.9.json',
            r'D:\code\music\qmx_reader\dataset_da\慢三六.json',
        ]
        for path in default_paths:
            if os.path.exists(path):
                args.input = path
                break
        if args.input is None:
            print("错误: 未找到默认JSON文件，请使用 --input 指定")
            exit(1)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    tensor_path = os.path.join(args.output, f"{base_name}_tensor_dataset.pt")

    # 1. 预处理
    if not args.no_preprocess:
        preprocess_json_file(
            args.input,
            args.output,
            sequence_size=args.sequence_size,
            subdivision=args.subdivision
        )
    else:
        print("\n[跳过] 预处理步骤")

    # 2. 验证
    if not args.no_verify:
        if os.path.exists(tensor_path):
            verify_tensor_dataset(args.input, tensor_path)
        else:
            print(f"\n[错误] 找不到tensor文件 {tensor_path}")
    else:
        print("\n[跳过] 验证步骤")
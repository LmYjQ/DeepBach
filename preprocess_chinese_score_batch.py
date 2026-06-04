"""
批量将简谱JSON格式转换为DeepBach模型所需的tensor格式

与preprocess_chinese_score.py的区别：
- 输入为文件夹，处理其中所有JSON文件
- 先遍历所有文件构建note2index和index2note映射
- 然后再逐个文件生成tensor
- 所有数据合并生成一个结果，通过sample_to_file区分每条样本来自哪个文件

输入格式 (简谱JSON):
  - value: 1-7 (do=1, re=2, ..., si=7) 表示音高
  - octave: 整数，表示八度偏移 (0=中央八度)
  - duration: 时值 (1=一拍, 0.5=半拍等)

输出格式 (DeepBach TensorDataset):
  - chorale_tensor: (N, 1, 32)  # N样本, 1声部, 32 ticks
  - metadata_tensor: (N, 1, 32, 2)  # tick + voice_id

用法:
  python preprocess_chinese_score_batch.py --input /path/to/json/folder --output ./preprocessed_data
"""
import json
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm

# 简谱音名到 MIDI 的映射 (首调，中央C为基准)
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
        current_tick: 总tick数
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


def build_global_note_mapping(all_midi_sequences):
    """
    从所有文件的MIDI序列构建全局的音符到索引映射

    Args:
        all_midi_sequences: 所有文件的MIDI序列列表

    Returns:
        note2index: 音符名到索引的映射
        index2note: 索引到音符名的映射
    """
    from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, REST_SYMBOL

    # 建立音符到索引的映射
    note_set = {SLUR_SYMBOL, START_SYMBOL, END_SYMBOL, REST_SYMBOL}

    # 遍历所有文件的音符，建立映射（跳过0=休止符）
    for midi_seq in all_midi_sequences:
        unique_pitches = set(midi_seq) - {0}  # 0 表示休止符，不作为音高
        for pitch in unique_pitches:
            note_name = f"{pitch}"
            note_set.add(note_name)

    # 创建映射表
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

    for pitch in sorted(note_set - set(special_symbols)):
        note_name = str(pitch)
        index2note[idx] = note_name
        note2index[note_name] = idx
        idx += 1

    print(f"全局音符表大小: {len(note2index)}")
    print(f"音符表示例: {list(note2index.items())[:10]}")

    return note2index, index2note


def create_tensor_dataset(midi_sequence, duration_sequence, is_articulated, note2index, index2note,
                          sequence_size=8, subdivision=4, voice_id=0):
    """
    使用预构建的全局映射创建DeepBach格式的TensorDataset

    Args:
        midi_sequence: MIDI音高序列
        duration_sequence: 时值序列
        is_articulated: 是否为音符起点的序列 (1=起点, 0=延续)
        note2index: 全局音符到索引的映射
        index2note: 全局索引到音符名的映射
        sequence_size: 每个样本的序列长度 (以拍为单位)
        subdivision: 每拍的tick数
        voice_id: 声部ID

    Returns:
        chorale_tensor: (N, 1, sequence_size * subdivision)
        metadata_tensor: (N, 1, sequence_size * subdivision, num_metadata)
    """
    # 创建序列
    from DatasetManager.helpers import SLUR_SYMBOL, REST_SYMBOL

    sequence_length_ticks = sequence_size * subdivision
    chorale_tensor_dataset = []
    metadata_tensor_dataset = []

    total_ticks = len(midi_sequence)

    # 预先计算所有metadata类型
    # IsPlaying: 1=正在演奏, 0=休止
    is_playing = (midi_sequence > 0).astype(int)

    # Tick: 0 到 subdivision-1 的位置循环
    tick_positions = np.arange(total_ticks) % subdivision

    # Mode: 1=D major (其他=0, 大调=1, 小调=2)
    MODE_MAJOR = 1
    mode_array = np.full(total_ticks, MODE_MAJOR)

    # Key: D major有2个升号，2 + 7 + 1 = 10
    KEY_D_MAJOR = 10  # 2 sharps = index 10
    key_array = np.full(total_ticks, KEY_D_MAJOR)

    # Fermata: 0=无延长记号
    FERMATA_NONE = 0
    fermata_array = np.full(total_ticks, FERMATA_NONE)

    # 滑动窗口采样
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
                idx = note2index.get(REST_SYMBOL, note2index.get('0'))
            elif articulated == 0:
                idx = note2index.get(SLUR_SYMBOL)
            else:
                note_name = str(pitch)
                idx = note2index.get(note_name, note2index.get(REST_SYMBOL))
            voice_tensor.append(idx)

        voice_tensor = np.array(voice_tensor).reshape(1, -1)  # (1, ticks)

        # 创建DeepBach格式的metadata (每种metadata一行)
        # metadata顺序: IsPlaying, Tick, Mode, Key, Fermata, voice_id
        window_is_playing = is_playing[start_tick:end_tick].reshape(1, -1)
        window_tick_pos = tick_positions[start_tick:end_tick].reshape(1, -1)
        window_mode = mode_array[start_tick:end_tick].reshape(1, -1)
        window_key = key_array[start_tick:end_tick].reshape(1, -1)
        window_fermata = fermata_array[start_tick:end_tick].reshape(1, -1)
        window_voice_id = np.full((1, sequence_length_ticks), voice_id)

        # 拼接所有metadata: (6, sequence_length_ticks)
        voice_metadata = np.concatenate([
            window_is_playing,
            window_tick_pos,
            window_mode,
            window_key,
            window_fermata,
            window_voice_id
        ], axis=0)

        # 转置为 (sequence_length_ticks, 6) 再unsqueeze为 (1, sequence_length_ticks, 6)
        voice_metadata = np.transpose(voice_metadata, (1, 0))  # (ticks, 6)
        voice_metadata = voice_metadata[np.newaxis, :, :]  # (1, ticks, 6)

        chorale_tensor_dataset.append(voice_tensor)
        metadata_tensor_dataset.append(voice_metadata)

    # 合并
    chorale_tensor = np.array(chorale_tensor_dataset)
    metadata_tensor = np.array(metadata_tensor_dataset)

    return chorale_tensor, metadata_tensor


def collect_midi_sequences_from_folder(folder_path, subdivision=4):
    """
    遍历文件夹中所有JSON文件，收集所有MIDI序列和文件信息

    Args:
        folder_path: JSON文件所在文件夹
        subdivision: 每拍tick数

    Returns:
        all_midi_sequences: 所有文件的MIDI序列列表
        file_info_list: 每个文件的信息列表 [(json_path, title, midi_seq, notes), ...]
    """
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    json_files = sorted(json_files)

    all_midi_sequences = []
    file_info_list = []

    print(f"发现 {len(json_files)} 个JSON文件")

    for json_file in tqdm(json_files, desc="扫描文件"):
        json_path = os.path.join(folder_path, json_file)

        try:
            data = load_json_dataset(json_path)
            title = data.get('title', 'unknown')
            notes = data.get('notes', [])

            # 过滤掉小节线等非音符数据
            notes = [n for n in notes if str(n.get('value', '')).lower() not in ['bar', 'space']]
            notes = [n for n in notes if n.get('value') is not None]

            midi_seq, dur_seq, is_articulated, _ = json_to_midi_sequence(notes, subdivision)
            all_midi_sequences.append(midi_seq)
            file_info_list.append({
                'path': json_path,
                'title': title,
                'filename': json_file,
                'midi_seq': midi_seq,
                'dur_seq': dur_seq,
                'is_articulated': is_articulated
            })
        except Exception as e:
            print(f"  警告: 处理文件 {json_file} 时出错: {e}")

    return all_midi_sequences, file_info_list


def preprocess_folder(folder_path, output_dir, sequence_size=8, subdivision=4, suffix=None):
    """
    批量预处理文件夹中的所有JSON文件，所有数据合并生成一个结果

    Args:
        folder_path: 输入文件夹路径
        output_dir: 输出目录
        sequence_size: 序列长度 (拍)
        subdivision: 每拍tick数
        suffix: 输出文件后缀 (默认从folder_path提取)
    """
    print("=" * 60)
    print(f"批量处理文件夹: {folder_path}")
    print("=" * 60)

    # 自动提取后缀: D:/code/music/qmx_reader/dataset_final8 -> final8
    if suffix is None:
        folder_name = os.path.basename(folder_path)
        if '_' in folder_name:
            parts = folder_name.split('_')
            suffix = '_'.join(parts[1:]) if len(parts) > 1 else folder_name
        else:
            suffix = folder_name

    print(f"后缀: {suffix}")

    # 1. 遍历所有文件，收集MIDI序列
    print("\n[1] 扫描文件，收集MIDI序列...")
    all_midi_sequences, file_info_list = collect_midi_sequences_from_folder(folder_path, subdivision)

    if not all_midi_sequences:
        print("错误: 未找到有效的JSON文件")
        return

    print(f"\n共收集到 {len(all_midi_sequences)} 个有效文件")

    # 2. 构建全局音符映射
    print("\n[2] 构建全局音符映射...")
    note2index, index2note = build_global_note_mapping(all_midi_sequences)

    # 3. 逐个文件生成tensor并合并
    print("\n[3] 逐个文件生成tensor并合并...")

    os.makedirs(output_dir, exist_ok=True)

    all_chorale_tensors = []
    all_metadata_tensors = []
    sample_to_file = []

    for file_info in tqdm(file_info_list, desc="生成tensor"):
        json_path = file_info['path']
        title = file_info['title']
        filename = file_info['filename']
        midi_seq = file_info['midi_seq']
        dur_seq = file_info['dur_seq']
        is_articulated = file_info['is_articulated']

        # 使用全局映射生成tensor
        chorale_tensor, metadata_tensor = create_tensor_dataset(
            midi_seq, dur_seq, is_articulated,
            note2index, index2note,
            sequence_size=sequence_size, subdivision=subdivision
        )

        all_chorale_tensors.append(chorale_tensor)
        all_metadata_tensors.append(metadata_tensor)

        # 记录每条样本对应的文件名
        num_samples = chorale_tensor.shape[0]
        sample_to_file.extend([filename] * num_samples)

    # 4. 合并所有tensor并保存
    print("\n[4] 合并所有tensor并保存...")

    combined_chorale = np.concatenate(all_chorale_tensors, axis=0)
    combined_metadata = np.concatenate(all_metadata_tensors, axis=0)

    # 输出文件名包含后缀: dataset_final8_combined_tensor_dataset.pt
    output_filename = f"dataset_{suffix}_combined_tensor_dataset.pt"
    combined_path = os.path.join(output_dir, output_filename)
    combined_dataset = {
        'note2index': note2index,
        'index2note': index2note,
        'sequence_size': sequence_size,
        'subdivision': subdivision,
        'chorale_tensor': combined_chorale,
        'metadata_tensor': combined_metadata,
        'file_count': len(file_info_list),
        'sample_to_file': sample_to_file,
        'metadata_info': {
            'types': ['IsPlaying', 'Tick', 'Mode', 'Key', 'Fermata', 'voice_id'],
            'Mode_values': {'other': 0, 'major': 1, 'minor': 2},
            'Key_values': {f'shifts_{i-7}': i for i in range(15)},
            'Key_fixed': 'D major (2 sharps, index=10)',
        },
    }

    torch.save(combined_dataset, combined_path)

    print(f"\n输出形状:")
    print(f"  combined_chorale_tensor: {combined_chorale.shape}")
    print(f"  combined_metadata_tensor: {combined_metadata.shape}")
    print(f"  sample_to_file长度: {len(sample_to_file)}")
    print(f"  合并tensor已保存至: {combined_path}")

    # 打印每个文件的样本数统计
    print("\n各文件样本数统计:")
    from collections import Counter
    file_counts = Counter(sample_to_file)
    for filename, count in sorted(file_counts.items()):
        print(f"  {filename}: {count} 样本")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='批量预处理简谱JSON为DeepBach格式')
    parser.add_argument('--input', '-i', default=r'D:/code/music/qmx_reader/dataset_final8',
                        help='输入文件夹路径')
    parser.add_argument('--output', '-o', default='./preprocessed_data',
                        help='输出目录')
    parser.add_argument('--sequence_size', '-s', type=int, default=8,
                        help='序列长度(拍), 默认8')
    parser.add_argument('--subdivision', '-d', type=int, default=8,
                        help='每拍tick数, 默认8 (32分音符)')
    parser.add_argument('--suffix', default=None,
                        help='输出文件后缀 (默认从输入文件夹名提取)')

    args = parser.parse_args()

    # 如果没有指定输入，使用默认文件夹
    if args.input is None:
        default_folders = [
            r'D:\code\music\qmx_reader\dataset_final8',
        ]
        for folder in default_folders:
            if os.path.exists(folder):
                args.input = folder
                break
        if args.input is None:
            print("错误: 未找到默认文件夹，请使用 --input 指定")
            exit(1)

    preprocess_folder(
        args.input,
        args.output,
        sequence_size=args.sequence_size,
        subdivision=args.subdivision,
        suffix=args.suffix
    )
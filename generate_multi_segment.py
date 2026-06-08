"""
多段生成脚本
基于简谱数据中的ban=1标记，固定某些音位，生成中间的音乐片段

用法:
  python generate_multi_segment.py --data <json_file> --pt <pt_file> --output generated.mid
"""
import argparse
import json
import torch
import numpy as np
import music21
from train_simple_notation import SimpleNotationDataset
from DeepBach.model_manager import DeepBach


def midi_to_value_octave(midi_pitch):
    """将MIDI pitch转换回JSON的value和octave格式"""
    if midi_pitch == 0:
        return '0', 0
    for octave in [1, 0, -1]:
        value = midi_pitch - 60 - octave
        if 0 <= value <= 7:
            return str(value), octave
    return str(midi_pitch - 60), 0


def idx_to_note_str(idx, index2note):
    """将索引转换为JSON格式的音符字符串"""
    note_str = index2note.get(idx, str(idx))
    if note_str.isdigit():
        midi = int(note_str)
        value, octave = midi_to_value_octave(midi)
        return f"{value}(o{octave})"
    elif note_str in ('REST', '0'):
        return '0'
    else:
        return note_str


def load_json_notes(json_path):
    """加载JSON格式的简谱音符数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    notes = data['notes']
    beats_per_bar = data.get('beatsPerBar', 4)

    print(f"加载JSON: {json_path}")
    print(f"  标题: {data.get('title', 'Unknown')}")
    print(f"  节拍: {beats_per_bar}")
    print(f"  音符数: {len(notes)}")

    return notes, beats_per_bar


def notes_to_ticks(notes, subdivision):
    """
    将音符列表转换为tick位置
    返回:
        ban_notes: list of (tick_pos, note_data) for notes with ban=1
        total_ticks: 总长度(ticks)
    """
    current_tick = 0
    ban_notes = []

    for note in notes:
        if note['ban'] == 1:
            ban_notes.append({
                'tick': current_tick,
                'id': note['id'],
                'value': note['value'],
                'octave': note.get('octave', 0),
                'duration': note['duration'],
                'dotted': note.get('dotted', False),
            })

        # duration是相对于beat的，所以要乘以subdivision
        current_tick += int(note['duration'] * subdivision)

    return ban_notes, current_tick


def find_segments(ban_notes, total_ticks, context_beats=2):
    """
    根据ban=1的音将曲子拆分成多段
    context_beats: 两端各保留多少beats作为上下文

    返回: list of (start_tick, end_tick, fixed_notes)
        其中 fixed_notes 是 [left_fixed, right_fixed] 的列表
        left_fixed/right_fixed 是 (tick, note_data) 或 None
    """
    segments = []
    context_ticks = context_beats * 8  # subdivision=8

    # 添加首尾标记
    all_boundaries = [0] + [n['tick'] for n in ban_notes] + [total_ticks]

    for i in range(len(all_boundaries) - 1):
        start_tick = all_boundaries[i]
        end_tick = all_boundaries[i + 1]

        # 按tick值查找左右固定的音，不是按数组索引
        left_fixed = None
        right_fixed = None
        for bn in ban_notes:
            if bn['tick'] == start_tick:
                left_fixed = bn
            if bn['tick'] == end_tick:
                right_fixed = bn

        # 如果区间太小，不需要生成
        if end_tick - start_tick <= 1:
            continue

        segments.append({
            'start': start_tick,
            'end': end_tick,
            'left_fixed': left_fixed,
            'right_fixed': right_fixed,
            'to_generate': (start_tick if left_fixed is None else left_fixed['tick'] + 1,
                           end_tick if right_fixed is None else right_fixed['tick'])
        })

    return segments


def create_tensor_from_json(notes, dataset, subdivision):
    """
    将JSON音符转换为DeepBach可用的chorale_tensor和metadata_tensor
    需要用dataset的note2index将MIDI pitch转为模型索引
    """
    num_voices = 1
    note2index = dataset.note2index_dicts[0]

    # 计算总tick数
    total_ticks = 0
    for note in notes:
        total_ticks += int(note['duration'] * subdivision)

    # 创建chorale tensor (num_voices, total_ticks)
    chorale_tensor = np.zeros((num_voices, total_ticks), dtype=np.int64)

    # 记录每个tick的元数据
    metadata_list = []

    current_tick = 0
    for note in notes:
        value = note['value']
        octave = note.get('octave', 0)
        dotted = note.get('dotted', False)

        # MIDI pitch计算 (简谱value是相对音高)
        if value.isdigit():
            base_pitch = int(value) + 60  # 默认中央C=60
            midi_pitch = base_pitch + octave
        elif value == '0':
            midi_pitch = 0  # 休止符
        else:
            midi_pitch = 60  # 默认

        # 转MIDI pitch为模型索引
        note_name = str(midi_pitch)
        note_idx = note2index.get(note_name, note2index.get('60', 0))

        # 设置chorale_tensor (这个音符占据的ticks)
        note_ticks = int(note['duration'] * subdivision)
        for t in range(note_ticks):
            if current_tick + t < total_ticks:
                chorale_tensor[0, current_tick + t] = note_idx

        # 记录metadata (tick, mode, key, fermata, voice_id)
        for t in range(note_ticks):
            metadata_list.append([
                (current_tick + t) % subdivision,  # tick in beat
                1,  # mode: major
                10,  # key: D major
                0,  # fermata
                0,  # voice_id
            ])

        current_tick += note_ticks

    # 转换为numpy
    metadata_tensor = np.array(metadata_list, dtype=np.int64)
    metadata_tensor = np.expand_dims(metadata_tensor, axis=0)  # (1, ticks, 5)

    return chorale_tensor, metadata_tensor, total_ticks


def generate_segment(deepbach, chorale_tensor, metadata_tensor,
                    start_tick, end_tick, fixed_notes, note2index,
                    temperature=1.0, num_iterations=500, batch_size=8):
    """
    生成一个片段

    fixed_notes: dict with 'left_fixed' and 'right_fixed', each is (tick, note_data) or None
    note2index: dict to convert MIDI pitch to model index
    """
    seq_length = end_tick - start_tick

    # 如果有固定的左音，从它之后开始生成；否则从头开始
    left_fixed = fixed_notes.get('left_fixed')
    right_fixed = fixed_notes.get('right_fixed')

    # 计算生成范围（相对位置，排除固定音）
    gen_start = 0  # 默认从头开始
    if left_fixed is not None:
        gen_start = left_fixed['tick'] - start_tick + 1

    gen_end = seq_length  # 默认到结尾
    if right_fixed is not None:
        gen_end = right_fixed['tick'] - start_tick

    # 相对位置
    rel_gen_start = gen_start
    rel_gen_end = gen_end

    print(f"  生成范围: 相对 {rel_gen_start} ~ {gen_end} (共 {gen_end - gen_start} ticks)")

    if rel_gen_start >= rel_gen_end:
        print(f"  跳过，无需要生成的区域")
        return None, tensor_chorale

    # 从chorale_tensor复制已有内容到tensor_chorale
    tensor_chorale = torch.from_numpy(chorale_tensor[:, start_tick:end_tick]).long()

    # metadata
    tensor_metadata = torch.from_numpy(metadata_tensor[:, start_tick:end_tick, :]).long()

    # 把固定音写入tensorChorale
    if left_fixed is not None:
        midi_pitch = int(left_fixed['value']) + 60 + left_fixed.get('octave', 0)
        note_idx = note2index.get(str(midi_pitch), note2index.get('60', 0))
        rel_pos = left_fixed['tick'] - start_tick
        if 0 <= rel_pos < seq_length:
            tensor_chorale[0, rel_pos] = note_idx
            print(f"  左固定: tick={left_fixed['tick']}, value={left_fixed['value']}, midi={midi_pitch}, idx={note_idx}")

    if right_fixed is not None:
        midi_pitch = int(right_fixed['value']) + 60 + right_fixed.get('octave', 0)
        note_idx = note2index.get(str(midi_pitch), note2index.get('60', 0))
        rel_pos = right_fixed['tick'] - start_tick
        if 0 <= rel_pos < seq_length:
            tensor_chorale[0, rel_pos] = note_idx
            print(f"  右固定: tick={right_fixed['tick']}, value={right_fixed['value']}, midi={midi_pitch}, idx={note_idx}")

    # 调用generation，随机初始化生成范围
    print(f"  生成前 tensor_chorale: {tensor_chorale.shape}")
    print(f"    索引: {tensor_chorale[0].tolist()}")
    print(f"    音符: {[idx_to_note_str(i, note2index) for i in tensor_chorale[0].tolist()]}")

    score, result_tensor, result_metadata = deepbach.generation(
        tensor_chorale=tensor_chorale,
        tensor_metadata=tensor_metadata,
        temperature=temperature,
        batch_size_per_voice=batch_size,
        num_iterations=num_iterations,
        sequence_length_ticks=seq_length,
        time_index_range_ticks=[rel_gen_start, rel_gen_end],
        random_init=True,
    )

    print(f"  生成后 result_tensor: {result_tensor.shape}")
    print(f"    索引: {result_tensor[0].tolist()}")
    print(f"    音符: {[idx_to_note_str(i, note2index) for i in result_tensor[0].tolist()]}")

    # generation结束后把固定音写回结果（因为random_init=True会在生成范围内随机初始化）
    if left_fixed is not None:
        rel_pos = left_fixed['tick'] - start_tick
        if 0 <= rel_pos < result_tensor.size(1):
            midi_pitch = int(left_fixed['value']) + 60 + left_fixed.get('octave', 0)
            note_idx = note2index.get(str(midi_pitch), note2index.get('60', 0))
            result_tensor[0, rel_pos] = note_idx

    if right_fixed is not None:
        rel_pos = right_fixed['tick'] - start_tick
        if 0 <= rel_pos < result_tensor.size(1):
            midi_pitch = int(right_fixed['value']) + 60 + right_fixed.get('octave', 0)
            note_idx = note2index.get(str(midi_pitch), note2index.get('60', 0))
            result_tensor[0, rel_pos] = note_idx

    return score, result_tensor


def main():
    parser = argparse.ArgumentParser(description='多段生成')
    parser.add_argument('--json', '-j', required=True,
                        help='简谱JSON文件路径')
    parser.add_argument('--data', '-d', required=True,
                        help='预处理生成的.pt文件路径')
    parser.add_argument('--subdivision', '-s', type=int, default=8,
                        help='每拍tick数 (默认8)')
    parser.add_argument('--output', '-o', default='generated_multi.mid',
                        help='输出MIDI文件路径')
    parser.add_argument('--models_dir', default='models',
                        help='模型目录 (默认models)')
    parser.add_argument('--num_iterations', '-n', type=int, default=500,
                        help='Gibbs采样迭代次数 (默认300)')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='batch大小 (默认8)')
    parser.add_argument('--temperature', '-t', type=float, default=1.0,
                        help='采样温度 (默认1.0)')
    parser.add_argument('--context_beats', '-c', type=int, default=2,
                        help='每段保留的上下文beats数 (默认2)')
    parser.add_argument('--note_embedding_dim', type=int, default=50,
                        help='音符嵌入维度 (默认50)')
    parser.add_argument('--meta_embedding_dim', type=int, default=25,
                        help='元数据嵌入维度 (默认25)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='LSTM层数 (默认2)')
    parser.add_argument('--lstm_hidden_size', type=int, default=128,
                        help='LSTM隐藏层大小 (默认128)')
    parser.add_argument('--dropout_lstm', type=float, default=0.5,
                        help='LSTM dropout (默认0.5)')
    parser.add_argument('--linear_hidden_size', type=int, default=128,
                        help='线性层隐藏大小 (默认128)')

    args = parser.parse_args()

    # 1. 加载JSON数据
    notes, beats_per_bar = load_json_notes(args.json)

    # 2. 找到所有ban=1的音
    ban_notes, total_ticks = notes_to_ticks(notes, args.subdivision)

    print(f"\n找到 {len(ban_notes)} 个板音 (ban=1):")
    total_gap_duration = 0
    for i, bn in enumerate(ban_notes):
        print(f"  [{i}] tick={bn['tick']}, value={bn['value']}, octave={bn['octave']}, duration={bn['duration']}")

    # 3. 计算中间间隔的duration加起来
    # 间隔是指ban=1音之间的间隔（不包括首尾）
    if len(ban_notes) > 1:
        for i in range(1, len(ban_notes)):
            gap_duration = ban_notes[i]['tick'] - ban_notes[i-1]['tick']
            total_gap_duration += gap_duration
            print(f"  间隔 {i}: {ban_notes[i-1]['tick']} -> {ban_notes[i]['tick']}, gap_ticks={gap_duration}")

    total_gap_beats = total_gap_duration / args.subdivision
    print(f"\n中间间隔总时长: {total_gap_duration} ticks = {total_gap_beats:.2f} beats")

    # 4. 拆分段落
    segments = find_segments(ban_notes, total_ticks, context_beats=args.context_beats)

    print(f"\n拆分段落 ({len(segments)} 段):")
    for i, seg in enumerate(segments):
        left = seg['left_fixed']
        right = seg['right_fixed']
        left_str = f"tick{left['tick']}(val={left['value']})" if left else "None"
        right_str = f"tick{right['tick']}(val={right['value']})" if right else "None"
        gap_ticks = seg['end'] - seg['start']
        gap_beats = gap_ticks / args.subdivision
        to_gen_ticks = seg['to_generate'][1] - seg['to_generate'][0]
        print(f"  段{i}: [{seg['start']},{seg['end']}] len={gap_beats:.2f}beats, "
              f"左右固定=[{left_str}, {right_str}], 需生成={to_gen_ticks} ticks")

    # 5. 加载数据集和模型
    print("\n加载数据集和模型...")
    dataset = SimpleNotationDataset(args.data, subdivision=args.subdivision)

    # 从数据路径提取模型后缀
    import os
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    parts = data_name.split('_')
    model_suffix = '_'.join(parts[1:]) if len(parts) > 1 else data_name

    deepbach = DeepBach(
        dataset=dataset,
        note_embedding_dim=args.note_embedding_dim,
        meta_embedding_dim=args.meta_embedding_dim,
        num_layers=args.num_layers,
        lstm_hidden_size=args.lstm_hidden_size,
        dropout_lstm=args.dropout_lstm,
        linear_hidden_size=args.linear_hidden_size,
        model_suffix=model_suffix,
        models_dir=args.models_dir,
    )

    # 打印模型配置信息
    print("\n=== 模型配置 ===")
    print(f"  模型后缀: {model_suffix}")
    print(f"  模型目录: {args.models_dir}")
    print(f"  音符嵌入维度: {args.note_embedding_dim}")
    print(f"  元数据嵌入维度: {args.meta_embedding_dim}")
    print(f"  LSTM层数: {args.num_layers}")
    print(f"  LSTM隐藏大小: {args.lstm_hidden_size}")
    print(f"  LSTM dropout: {args.dropout_lstm}")
    print(f"  线性层隐藏大小: {args.linear_hidden_size}")

    # 检查模型文件是否存在
    import os
    expected_model_file = os.path.join(args.models_dir, f"voicemodel_{model_suffix}_0.pt")
    print(f"\n期望的模型文件: {expected_model_file}")
    print(f"文件存在: {os.path.exists(expected_model_file)}")

    # 列出models目录下所有文件
    if os.path.exists(args.models_dir):
        print(f"\n{args.models_dir} 目录下的文件:")
        for f in os.listdir(args.models_dir):
            print(f"  {f}")
    else:
        print(f"\n警告: 模型目录 {args.models_dir} 不存在!")

    print("\n开始加载模型...")
    deepbach.load()
    print("模型加载完成!")

    # 6. 创建tensor数据
    print("\n创建tensor数据...")
    chorale_tensor, metadata_tensor, total_ticks = create_tensor_from_json(notes, dataset, args.subdivision)
    print(f"  chorale_tensor: {chorale_tensor.shape}")
    print(f"  metadata_tensor: {metadata_tensor.shape}")
    print(f"  total_ticks: {total_ticks}")

    # 7. 级联生成：上一段的输出直接作为下一段的上下文
    print("\n开始生成...")
    note2index = dataset.note2index_dicts[0]
    index2note = dataset.index2note_dicts[0]
    print(f"  note2index 大小: {len(note2index)}")
    print(f"  index2note 示例: {dict(list(index2note.items())[:20])}")

    for i, seg in enumerate(segments):
        print(f"\n=== 段 {i} ===")

        left_fixed = seg['left_fixed']
        right_fixed = seg['right_fixed']

        # 级联生成：对于后续段，使用更新后的chorale_tensor
        print(f"  级联上下文: chorale_tensor[{seg['start']}:{seg['end']}] = {chorale_tensor[0, seg['start']:seg['end']].tolist()}")
        score, result_tensor = generate_segment(
            deepbach,
            chorale_tensor,
            metadata_tensor,
            seg['start'],
            seg['end'],
            {'left_fixed': left_fixed, 'right_fixed': right_fixed},
            note2index,
            temperature=args.temperature,
            num_iterations=args.num_iterations,
            batch_size=args.batch_size,
        )

        # 级联：把生成结果写回chorale_tensor，供下一段使用
        seg_len = seg['end'] - seg['start']
        if result_tensor is not None:
            # 只更新本段的生成范围（排除左右固定音）
            gen_start_abs = seg['start']
            gen_end_abs = seg['end']

            if left_fixed is not None:
                gen_start_abs = left_fixed['tick'] + 1
            if right_fixed is not None:
                gen_end_abs = right_fixed['tick']

                       # 将结果写回chorale_tensor
            for abs_tick in range(gen_start_abs, gen_end_abs):
                rel_tick = abs_tick - seg['start']
                if 0 <= rel_tick < result_tensor.size(1):
                    chorale_tensor[0, abs_tick] = result_tensor[0, rel_tick].item()

        # 打印上下文中的固定音
        left_in_ctx = left_fixed is not None
        right_in_ctx = right_fixed is not None
        print(f"  上下文可见: 左固定={left_in_ctx}, 右固定={right_in_ctx}")
        print(f"  生成完成")

    # 8. 合并结果并保存
    print("\n合并结果...")

    # 级联生成完成后，chorale_tensor已经被更新为最终结果
    # 但ban=1的固定音可能还是旧值，需要写回去
    print("\n写回固定音...")
    for i, seg in enumerate(segments):
        left_fixed = seg['left_fixed']
        right_fixed = seg['right_fixed']

        if left_fixed is not None:
            midi_pitch = int(left_fixed['value']) + 60 + left_fixed.get('octave', 0)
            note_idx = note2index.get(str(midi_pitch), note2index.get('60', 0))
            chorale_tensor[0, left_fixed['tick']] = note_idx
            print(f"  写回左固定: tick={left_fixed['tick']}, value={left_fixed['value']}, idx={note_idx}")

        if right_fixed is not None:
            midi_pitch = int(right_fixed['value']) + 60 + right_fixed.get('octave', 0)
            note_idx = note2index.get(str(midi_pitch), note2index.get('60', 0))
            chorale_tensor[0, right_fixed['tick']] = note_idx
            print(f"  写回右固定: tick={right_fixed['tick']}, value={right_fixed['value']}, idx={note_idx}")

    # 转回torch tensor
    final_torch = torch.from_numpy(chorale_tensor).long()

    # 创建metadata tensor
    metadata_torch = torch.from_numpy(metadata_tensor).long()

    # 转换为music21 score
    score = dataset.tensor_to_score(final_torch, metadata_torch[:, :, 3])  # fermata index=3

    # 保存
    print(f"\n保存到: {args.output}")
    mf = music21.midi.translate.music21ObjectToMidiFile(score)
    mf.open(args.output, 'wb')
    mf.write()
    mf.close()

    print("生成完成!")


if __name__ == '__main__':
    main()
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
                    temperature=1.0, num_iterations=300, batch_size=8):
    """
    生成一个片段

    fixed_notes: dict with 'left_fixed' and 'right_fixed', each is (tick, note_data) or None
    note2index: dict to convert MIDI pitch to model index
    """
    seq_length = end_tick - start_tick

    # 创建完整的序列tensor
    if chorale_tensor is None:
        tensor_chorale = deepbach.dataset.empty_score_tensor(seq_length)
    else:
        tensor_chorale = torch.from_numpy(chorale_tensor[:, start_tick:end_tick]).long()

    # metadata
    tensor_metadata = torch.from_numpy(metadata_tensor[:, start_tick:end_tick, :]).long()

    # 找出需要固定的音的位置
    left_fixed = fixed_notes.get('left_fixed')
    right_fixed = fixed_notes.get('right_fixed')

    # 计算生成范围（排除两端固定的音）
    gen_start = start_tick
    gen_end = end_tick

    if left_fixed is not None:
        gen_start = left_fixed['tick'] + 1
        # 将左固定音设置到tensor_chorale中 (相对位置)
        midi_pitch = int(left_fixed['value']) + 60 + left_fixed.get('octave', 0)
        note_idx = note2index.get(str(midi_pitch), note2index.get('60', 0))
        rel_pos = left_fixed['tick'] - start_tick
        if 0 <= rel_pos < seq_length:
            tensor_chorale[0, rel_pos] = note_idx

    if right_fixed is not None:
        gen_end = right_fixed['tick']
        # 将右固定音设置到tensor_chorale中 (相对位置)
        midi_pitch = int(right_fixed['value']) + 60 + right_fixed.get('octave', 0)
        note_idx = note2index.get(str(midi_pitch), note2index.get('60', 0))
        rel_pos = right_fixed['tick'] - start_tick
        if 0 <= rel_pos < seq_length:
            tensor_chorale[0, rel_pos] = note_idx

    # 相对位置
    rel_gen_start = gen_start - start_tick
    rel_gen_end = gen_end - start_tick

    print(f"  生成范围: tick {gen_start} ~ {gen_end} (相对 {rel_gen_start} ~ {rel_gen_end})")

    if rel_gen_start >= rel_gen_end:
        print(f"  跳过，无需要生成的区域")
        return None, tensor_chorale

    # 调用generation，只重生成中间的音
    score, result_tensor, result_metadata = deepbach.generation(
        tensor_chorale=tensor_chorale,
        tensor_metadata=tensor_metadata,
        temperature=temperature,
        batch_size_per_voice=batch_size,
        num_iterations=num_iterations,
        sequence_length_ticks=seq_length,
        time_index_range_ticks=[rel_gen_start, rel_gen_end],
        random_init=False,
    )

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
    parser.add_argument('--num_iterations', '-n', type=int, default=300,
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

    print("加载模型...")
    deepbach.load()

    # 6. 创建tensor数据
    print("\n创建tensor数据...")
    chorale_tensor, metadata_tensor, total_ticks = create_tensor_from_json(notes, dataset, args.subdivision)
    print(f"  chorale_tensor: {chorale_tensor.shape}")
    print(f"  metadata_tensor: {metadata_tensor.shape}")
    print(f"  total_ticks: {total_ticks}")

    # 7. 对每一段调用generation
    print("\n开始生成...")
    generated_ticks = []
    note2index = dataset.note2index_dicts[0]

    for i, seg in enumerate(segments):
        print(f"\n=== 段 {i} ===")

        # 对于第一段，从随机初始化开始；对于后续段，基于上一段结果
        # 这里简化处理：每次都传入完整的tensor_chorale
        score, result_tensor = generate_segment(
            deepbach,
            chorale_tensor,
            metadata_tensor,
            seg['start'],
            seg['end'],
            {'left_fixed': seg['left_fixed'], 'right_fixed': seg['right_fixed']},
            note2index,
            temperature=args.temperature,
            num_iterations=args.num_iterations,
            batch_size=args.batch_size,
        )

        # 保存生成的片段到总结果
        gen_start, gen_end = seg['to_generate']
        for tick in range(gen_start, gen_end):
            if tick < result_tensor.size(1):
                generated_ticks.append((tick, result_tensor[0, tick - seg['start']].item()))

        print(f"  生成完成")

    # 8. 合并结果并保存
    print("\n合并结果...")

    # 创建完整的chorale tensor，将ban=1的音保留，其余用生成的填充
    final_tensor = chorale_tensor.copy()

    for tick, pitch in generated_ticks:
        if tick < final_tensor.shape[1]:
            final_tensor[0, tick] = pitch

    # 转回torch tensor
    final_torch = torch.from_numpy(final_tensor).long()

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
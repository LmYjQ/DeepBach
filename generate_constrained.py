"""
约束生成脚本
基于简谱数据中的ban=1标记，固定某些音位，一次性生成全部音乐

用法:
  python generate_constrained.py --json<json_file> --data <pt_file> --model <pt_model> --output generated.mid
"""
import argparse
import json
import os
import torch
import numpy as np
import music21
from train_simple_notation import SimpleNotationDataset
from DeepBach.model_manager import DeepBach
from DeepBach.voice_model import parse_model_filename


def midi_to_value_octave(midi_pitch):
    """将MIDI pitch转换回JSON的value和octave格式"""
    if midi_pitch == 0:
        return '0', 0

    SOLFEGE_TO_MIDI_BASE = {
        1: 60, 2: 62, 3: 64, 4: 65, 5: 67, 6: 69, 7: 71
    }

    for octave in [1, 0, -1]:
        remaining = midi_pitch - octave * 12
        for value, base_midi in SOLFEGE_TO_MIDI_BASE.items():
            if remaining == base_midi:
                return str(value), octave

    return str(midi_pitch), 0


def idx_to_note_str(idx, index2note):
    """将索引转换为JSON格式的音符字符串"""
    note_str = index2note.get(idx, str(idx))
    if note_str.isdigit():
        midi = int(note_str)
        value, octave = midi_to_value_octave(midi)
        return f"{value}(o{octave})"
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

        current_tick += int(note['duration'] * subdivision)

    return ban_notes, current_tick


def create_tensor_from_json(notes, dataset, subdivision):
    """
    将JSON音符转换为DeepBach可用的chorale_tensor和metadata_tensor
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

    SOLFEGE_TO_MIDI_BASE = {
        1: 60, 2: 62, 3: 64, 4: 65, 5: 67, 6: 69, 7: 71
    }

    current_tick = 0
    for note in notes:
        value = note['value']
        octave = note.get('octave', 0)
        dotted = note.get('dotted', False)

        # MIDI pitch计算
        if value.isdigit() and int(value) in SOLFEGE_TO_MIDI_BASE:
            base_midi = SOLFEGE_TO_MIDI_BASE[int(value)]
            midi_pitch = base_midi + (octave * 12)
        elif value == '0':
            midi_pitch = 0  # 休止符
        else:
            midi_pitch = 60  # 默认

        # 转MIDI pitch为模型索引
        note_name = str(midi_pitch)
        note_idx = note2index.get(note_name, note2index.get('60', 0))

        # 设置chorale_tensor
        note_ticks = int(note['duration'] * subdivision)
        for t in range(note_ticks):
            if current_tick + t < total_ticks:
                chorale_tensor[0, current_tick + t] = note_idx

        # 记录metadata
        for t in range(note_ticks):
            metadata_list.append([
                (current_tick + t) % subdivision,  # tick in beat
                1,  # mode: major
                10,  # key: D major
                0,  # fermata
                0,  # voice_id
            ])

        current_tick += note_ticks

    metadata_tensor = np.array(metadata_list, dtype=np.int64)
    metadata_tensor = np.expand_dims(metadata_tensor, axis=0)  # (1, ticks, 5)

    return chorale_tensor, metadata_tensor, total_ticks


def main():
    parser = argparse.ArgumentParser(description='约束生成 - 固定板音，一次性生成')
    parser.add_argument('--json', '-j', required=True,
                        help='简谱JSON文件路径')
    parser.add_argument('--data', '-d', required=True,
                        help='预处理生成的.pt文件路径')
    parser.add_argument('--model', '-m', required=True,
                        help='模型文件路径')
    parser.add_argument('--subdivision', '-s', type=int, default=8,
                        help='每拍tick数 (默认8)')
    parser.add_argument('--output', '-o', default='generated_constrained.mid',
                        help='输出MIDI文件路径')
    parser.add_argument('--num_iterations', '-n', type=int, default=500,
                        help='Gibbs采样迭代次数 (默认500)')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='batch大小 (默认8)')
    parser.add_argument('--temperature', '-t', type=float, default=1.0,
                        help='采样温度 (默认1.0)')

    args = parser.parse_args()

    # 从模型文件名解析参数
    model_filename = os.path.basename(args.model)
    model_params = parse_model_filename(model_filename)
    if model_params is None:
        raise ValueError(f"无法从模型文件名解析参数: {model_filename}")
    print(f"\n=== 从模型文件解析的参数 ===")
    print(f"  note_embedding_dim: {model_params['note_embedding_dim']}")
    print(f"  meta_embedding_dim: {model_params['meta_embedding_dim']}")
    print(f"  lstm_hidden_size: {model_params['lstm_hidden_size']}")
    print(f"  num_layers: {model_params['num_layers']}")
    print(f"  dropout_lstm: {model_params['dropout_lstm']}")
    print(f"  hidden_size_linear: {model_params['hidden_size_linear']}")

    # 1. 加载JSON数据
    notes, beats_per_bar = load_json_notes(args.json)

    # 2. 找到所有ban=1的音作为固定位置
    ban_notes, total_ticks = notes_to_ticks(notes, args.subdivision)

    print(f"\n找到 {len(ban_notes)} 个板音 (ban=1):")
    for i, bn in enumerate(ban_notes):
        print(f"  [{i}] tick={bn['tick']}, value={bn['value']}, octave={bn['octave']}")

    # 提取固定位置的tick列表
    fixed_ticks = [bn['tick'] for bn in ban_notes]
    print(f"\n固定位置列表: {fixed_ticks}")

    # 3. 加载数据集和模型
    print("\n加载数据集和模型...")
    dataset = SimpleNotationDataset(args.data, subdivision=args.subdivision)

    deepbach = DeepBach(
        dataset=dataset,
        note_embedding_dim=model_params['note_embedding_dim'],
        meta_embedding_dim=model_params['meta_embedding_dim'],
        num_layers=model_params['num_layers'],
        lstm_hidden_size=model_params['lstm_hidden_size'],
        dropout_lstm=model_params['dropout_lstm'],
        linear_hidden_size=model_params['hidden_size_linear'],
        model_suffix="",
        models_dir=os.path.dirname(args.model) or "models",
    )

    print(f"\n模型文件: {args.model}")
    print(f"文件存在: {os.path.exists(args.model)}")

    print("\n加载模型...")
    deepbach.load(model_path=args.model)
    print("模型加载完成!")

    # 4. 创建tensor数据
    print("\n创建tensor数据...")
    chorale_tensor, metadata_tensor, total_ticks = create_tensor_from_json(
        notes, dataset, args.subdivision)
    print(f"  chorale_tensor: {chorale_tensor.shape}")
    print(f"  metadata_tensor: {metadata_tensor.shape}")
    print(f"  total_ticks: {total_ticks}")

    note2index = dataset.note2index_dicts[0]
    index2note = dataset.index2note_dicts[0]

    # 5. 一次性生成
    print("\n开始生成...")
    print(f"  生成范围: [0, {total_ticks}]")
    print(f"  固定位置: {fixed_ticks}")
    print(f"  迭代次数: {args.num_iterations}")
    print(f"  采样温度: {args.temperature}")

    tensor_chorale = torch.from_numpy(chorale_tensor).long()
    tensor_metadata = torch.from_numpy(metadata_tensor).long()

    print(f"\n生成前 tensor_chorale: {tensor_chorale.shape}")
    print(f"  索引: {tensor_chorale[0].tolist()}")

    score, result_tensor, result_metadata = deepbach.generation(
        tensor_chorale=tensor_chorale,
        tensor_metadata=tensor_metadata,
        temperature=args.temperature,
        batch_size_per_voice=args.batch_size,
        num_iterations=args.num_iterations,
        sequence_length_ticks=total_ticks,
        time_index_range_ticks=[0, total_ticks],
        time_index_list_ticks=fixed_ticks,  # 新参数：固定这些位置
        random_init=False,
    )

    print(f"\n生成后 result_tensor: {result_tensor.shape}")
    print(f"  索引: {result_tensor[0].tolist()}")

    # 6. 校验固定音是否保持正确
    print("\n校验固定音...")
    SOLFEGE_TO_MIDI_BASE = {1: 60, 2: 62, 3: 64, 4: 65, 5: 67, 6: 69, 7: 71}

    all_correct = True
    for bn in ban_notes:
        value = int(bn['value'])
        octave = bn.get('octave', 0)
        midi_pitch = SOLFEGE_TO_MIDI_BASE[value] + (octave * 12)
        expected_idx = note2index.get(str(midi_pitch), note2index.get('60', 0))
        actual_idx = result_tensor[0, bn['tick']].item()
        is_correct = actual_idx == expected_idx
        all_correct = all_correct and is_correct
        status = "✓" if is_correct else "✗"
        print(f"  {status} tick={bn['tick']}, expected idx={expected_idx}, actual idx={actual_idx}")

    if all_correct:
        print("  校验通过：所有固定音均正确")
    else:
        print("  警告：部分固定音不一致")

    # 7. 转换为music21 score并保存
    print("\n转换为music21 score...")
    final_torch = result_tensor.long()
    metadata_torch = result_metadata.long()

    score = dataset.tensor_to_score(final_torch, metadata_torch[:, :, 3])  # fermata index=3

    print(f"\n保存到: {args.output}")
    mf = music21.midi.translate.music21ObjectToMidiFile(score)
    mf.open(args.output, 'wb')
    mf.write()
    mf.close()

    print("生成完成!")


if __name__ == '__main__':
    main()
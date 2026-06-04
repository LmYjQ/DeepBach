"""
简谱DeepBach生成脚本
使用训练好的模型生成乐谱

用法:
  python generate_simple_notation.py --data <pt_file> --output melody.mid
  python generate_simple_notation.py --data <pt_file> --load --output melody.mid
"""
import argparse
import torch
import music21
import numpy as np
from datetime import datetime
from DatasetManager.metadata import (
    TickMetadata,
    FermataMetadata,
    KeyMetadata,
    ModeMetadata,
)
from train_simple_notation import SimpleNotationDataset
from DeepBach.model_manager import DeepBach


def main():
    parser = argparse.ArgumentParser(description='简谱DeepBach生成')
    parser.add_argument('--data', '-d', required=True,
                        help='预处理生成的.pt文件路径')
    parser.add_argument('--subdivision', '-s', type=int, default=8,
                        help='每拍tick数 (默认8)')
    parser.add_argument('--output', '-o', default='generated_score.mid',
                        help='输出MIDI文件路径')
    parser.add_argument('--load', action='store_true',
                        help='加载已有模型')
    parser.add_argument('--models_dir', default='models',
                        help='模型目录 (默认models)')
    parser.add_argument('--model_suffix', default='',
                        help='模型后缀 (默认从数据路径提取)')
    parser.add_argument('--note_embedding_dim', type=int, default=100,
                        help='音符嵌入维度')
    parser.add_argument('--meta_embedding_dim', type=int, default=50,
                        help='元数据嵌入维度')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='LSTM层数')
    parser.add_argument('--lstm_hidden_size', type=int, default=200,
                        help='LSTM隐藏层大小')
    parser.add_argument('--dropout_lstm', type=float, default=0.2,
                        help='LSTM dropout')
    parser.add_argument('--linear_hidden_size', type=int, default=200,
                        help='线性层隐藏大小')
    parser.add_argument('--num_iterations', '-n', type=int, default=500,
                        help='Gibbs采样迭代次数 (默认500)')
    parser.add_argument('--sequence_length_ticks', type=int, default=64,
                        help='生成序列长度 (ticks，默认64)')
    parser.add_argument('--batch_size_per_voice', type=int, default=8,
                        help='每声部batch大小 (默认8)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='采样温度 (默认1.0)')
    parser.add_argument('--repeat', type=int, default=1,
                        help='重复生成次数 (默认1)')

    args = parser.parse_args()

    # 自动提取模型后缀
    import os
    data_path = args.data
    data_name = os.path.splitext(os.path.basename(data_path))[0]
    if '_' in data_name:
        parts = data_name.split('_')
        auto_suffix = '_'.join(parts[1:]) if len(parts) > 1 else data_name
    else:
        auto_suffix = data_name

    model_suffix = args.model_suffix if args.model_suffix else auto_suffix
    print(f"模型后缀: {model_suffix}")
    print(f"模型目录: {args.models_dir}")

    # 加载数据集
    dataset = SimpleNotationDataset(args.data, subdivision=args.subdivision)

    # 创建模型
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

    if args.load:
        print("加载已有模型...")
        deepbach.load()
    else:
        print("创建新模型 (随机初始化)...")

    print("\n开始生成...")
    print(f"  迭代次数: {args.num_iterations}")
    print(f"  序列长度: {args.sequence_length_ticks} ticks = {args.sequence_length_ticks // args.subdivision} beats")
    print(f"  温度: {args.temperature}")

    # 创建固定metadata用于生成 (D major)
    # metadata格式: [Tick, Mode, Key, Fermata, voice_id]
    seq_len = args.sequence_length_ticks
    tick_positions = (np.arange(seq_len) % args.subdivision).reshape(-1, 1)
    mode_vals = np.full((seq_len, 1), 1)  # D major = major
    key_vals = np.full((seq_len, 1), 10)   # D major index
    fermata_vals = np.zeros((seq_len, 1), dtype=np.int64)
    voice_id_vals = np.zeros((seq_len, 1), dtype=np.int64)  # voice_id = 0

    # metadata: (seq_len, num_metadata) -> (1, seq_len, num_metadata)
    tensor_metadata = np.concatenate([tick_positions, mode_vals, key_vals, fermata_vals, voice_id_vals], axis=1)
    tensor_metadata = np.expand_dims(tensor_metadata, axis=0)  # add voice dim
    tensor_metadata = torch.from_numpy(tensor_metadata).long()

    # 生成
    score, tensor_chorale, tensor_metadata = deepbach.generation(
        temperature=args.temperature,
        batch_size_per_voice=args.batch_size_per_voice,
        num_iterations=args.num_iterations,
        sequence_length_ticks=args.sequence_length_ticks,
        tensor_metadata=tensor_metadata,
        random_init=True
    )

    # 保存为MIDI
    print(f"\n保存到: {args.output}")
    mf = music21.midi.translate.music21ObjectToMidiFile(score)
    mf.open(args.output, 'wb')
    mf.write()
    mf.close()

    print("生成完成!")

    # 可选：显示乐谱
    # score.show()


if __name__ == '__main__':
    main()
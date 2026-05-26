"""
简谱DeepBach训练脚本
从预处理生成的.pt文件加载数据，调用DeepBach模型进行训练

用法:
  python train_simple_notation.py --data <pt_file> --train 15
"""
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader
from DatasetManager.metadata import (
    TickMetadata,
    FermataMetadata,
    KeyMetadata,
    ModeMetadata,
)
from DeepBach.model_manager import DeepBach


class SimpleNotationDataset:
    """简谱数据集封装，从预处理的.pt文件加载"""
    def __init__(self, pt_filepath, subdivision=8):
        self.pt_filepath = pt_filepath
        self.data = torch.load(pt_filepath, map_location='cpu', weights_only=False)

        self.chorale_tensor = self.data['chorale_tensor']
        self.metadata_tensor = self.data['metadata_tensor']
        self.note2index_dicts = [self.data['note2index']]
        self.index2note_dicts = [self.data['index2note']]
        self.sequence_size = self.data['sequence_size']
        self.sequences_size = self.data['sequence_size']  # alias for DeepBach
        self.subdivision = subdivision
        self.num_voices = 1

        # Metadata types - 必须与 metadata_tensor[:,:,:,[1,2,3,4]] 的列顺序一致
        # [Tick, Mode, Key, Fermata] (跳过IsPlaying和voice_id)
        self.metadatas = [
            TickMetadata(subdivision=subdivision),  # index 0 after extraction (original index 1)
            ModeMetadata(),                         # index 1 after extraction (original index 2)
            KeyMetadata(),                          # index 2 after extraction (original index 3)
            FermataMetadata(),                      # index 3 after extraction (original index 4)
            # voice_id 是第5个embedding (index 4)
        ]

        print(f"加载数据集: {pt_filepath}")
        print(f"  chorale_tensor: {self.chorale_tensor.shape}")
        print(f"  metadata_tensor: {self.metadata_tensor.shape}")
        print(f"  音符表大小: {len(self.note2index_dicts[0])}")

    def __repr__(self):
        return f"SimpleNotationDataset({self.pt_filepath})"

    def data_loaders(self, batch_size, split=(0.85, 0.10)):
        """返回train/val/test三个数据加载器"""
        assert sum(split) < 1

        dataset = self.tensor_dataset
        num_examples = len(dataset)
        a, b = split

        train_dataset = TensorDataset(*dataset[: int(a * num_examples)])
        val_dataset = TensorDataset(*dataset[int(a * num_examples):
                                           int((a + b) * num_examples)])
        eval_dataset = TensorDataset(*dataset[int((a + b) * num_examples):])

        train_dl = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        eval_dl = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        return train_dl, val_dl, eval_dl

    @property
    def tensor_dataset(self):
        # chorale_tensor: (N, 1, ticks)
        # metadata_tensor: (N, 1, ticks, 6) -> (N, ticks, 6) after removing voice dim
        chorale = torch.from_numpy(self.chorale_tensor).long()
        metadata = torch.from_numpy(self.metadata_tensor).long()

        # 移除voice维度 metadata: (N, 1, ticks, 6) -> (N, ticks, 6)
        metadata = metadata.squeeze(1)

        # VoiceModel期望 metadata: (N, ticks, num_metadata)
        # 但预处理输出是 [IsPlaying, Tick, Mode, Key, Fermata, voice_id]
        # 而DeepBach训练使用 [Tick, Mode, Key, Fermata] (不含IsPlaying和voice_id)

        # 我们需要把metadata_tensor转换为VoiceModel期望的格式
        # VoiceModel.train_model() 调用 dataset.data_loaders() 返回的tensor_metadata
        # 然后 preprocess_metas() 会索引 main_voice_index

        # 问题是：我们的metadata_tensor是(N, ticks, 6)，没有voice维度
        # 而VoiceModel期望 (batch, num_voices, ticks, num_metadata)

        # 解决方案：重新组织metadata_tensor，添加voice维度
        # metadata_tensor: (N, ticks, 6) -> (N, 1, ticks, 6)
        metadata = metadata.unsqueeze(1)

        # 只取 [Tick, Mode, Key, Fermata, voice_id] (索引1,2,3,4,5)，跳过IsPlaying
        # 这是DeepBach实际使用的metadata
        metadata = metadata[:, :, :, [1, 2, 3, 4, 5]]  # (N, 1, ticks, 5)

        # 注意：这里metadata[:, 0, :, :] 会返回每个样本的metadata
        # 但data_loaders返回的是整个dataset
        return TensorDataset(chorale, metadata)

    def extract_score_tensor_with_padding(self, tensor_score, start_tick, end_tick):
        """padding提取score tensor"""
        from DatasetManager.helpers import START_SYMBOL, END_SYMBOL

        seq_len = tensor_score.size(1)
        pad_left = -start_tick if start_tick < 0 else 0
        pad_right = end_tick - seq_len if end_tick > seq_len else 0

        if pad_left > 0 or pad_right > 0:
            num_voices = tensor_score.size(0)
            start_idx = self.note2index_dicts[0].get(START_SYMBOL, 0)
            end_idx = self.note2index_dicts[0].get(END_SYMBOL, 0)
            start_pad = tensor_score.new_full((num_voices, pad_left), start_idx)
            end_pad = tensor_score.new_full((num_voices, pad_right), end_idx)
            tensor_score = torch.cat([start_pad, tensor_score, end_pad], dim=1)
            start_tick = 0
            end_tick = pad_left + seq_len + pad_right  # actual end in padded tensor

        return tensor_score[:, start_tick:end_tick]

    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick):
        """padding提取metadata tensor"""
        seq_len = tensor_metadata.size(1)
        pad_left = -start_tick if start_tick < 0 else 0
        pad_right = end_tick - seq_len if end_tick > seq_len else 0

        if pad_left > 0 or pad_right > 0:
            num_voices = tensor_metadata.size(0)
            num_metadatas = tensor_metadata.size(2)
            start_pad = tensor_metadata.new_zeros((num_voices, pad_left, num_metadatas))
            end_pad = tensor_metadata.new_zeros((num_voices, pad_right, num_metadatas))
            tensor_metadata = torch.cat([start_pad, tensor_metadata, end_pad], dim=1)

        return tensor_metadata[start_tick:end_tick]

    def empty_score_tensor(self, score_length):
        """返回用START_SYMBOL初始化的tensor"""
        from DatasetManager.helpers import START_SYMBOL
        start_idx = self.note2index_dicts[0].get(START_SYMBOL, 0)
        indices = torch.full((self.num_voices, score_length),
                           start_idx,
                           dtype=torch.long)
        return indices

    def random_score_tensor(self, score_length):
        """返回随机初始化的tensor"""
        import random
        result = []
        for voice_id in range(self.num_voices):
            note2index = self.note2index_dicts[voice_id]
            num_notes = len(note2index)
            voice_tensor = torch.tensor(
                [random.randint(0, num_notes - 1) for _ in range(score_length)],
                dtype=torch.long
            )
            result.append(voice_tensor)
        return torch.stack(result, dim=0)

    def tensor_to_score(self, tensor_score, fermata_tensor=None):
        """将tensor转换为music21 Score"""
        from music21 import stream, note as m21note, duration
        from DatasetManager.helpers import SLUR_SYMBOL, REST_SYMBOL, START_SYMBOL, END_SYMBOL

        index2note = self.index2note_dicts[0]

        s = stream.Stream()
        p = stream.Part()

        current_duration = 0
        for tick_idx in range(tensor_score.size(0)):
            note_idx = tensor_score[tick_idx].item()
            note_name = index2note.get(note_idx, '')

            if note_name == SLUR_SYMBOL:
                current_duration += 1
                continue
            elif note_name in (REST_SYMBOL, '0'):
                if current_duration > 0:
                    n = m21note.Rest()
                    n.duration = duration.Duration(current_duration / self.subdivision)
                    p.append(n)
                    current_duration = 0
                n = m21note.Rest()
                n.duration = duration.Duration(1 / self.subdivision)
                p.append(n)
            elif note_name in (START_SYMBOL, END_SYMBOL):
                continue
            else:
                if current_duration > 0:
                    r = m21note.Rest()
                    r.duration = duration.Duration(current_duration / self.subdivision)
                    p.append(r)
                    current_duration = 0

                midi_pitch = int(note_name) if note_name.isdigit() else 60
                n = m21note.Note()
                n.pitch.ps = midi_pitch
                n.duration = duration.Duration(1 / self.subdivision)
                p.append(n)

        s.append(p)
        return s


def main():
    parser = argparse.ArgumentParser(description='简谱DeepBach训练')
    parser.add_argument('--data', '-d', required=True,
                        help='预处理生成的.pt文件路径')
    parser.add_argument('--subdivision', '-s', type=int, default=8,
                        help='每拍tick数 (默认8)')
    parser.add_argument('--note_embedding_dim', type=int, default=100,
                        help='音符嵌入维度 (默认100)')
    parser.add_argument('--meta_embedding_dim', type=int, default=50,
                        help='元数据嵌入维度 (默认50)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='LSTM层数 (默认2)')
    parser.add_argument('--lstm_hidden_size', type=int, default=200,
                        help='LSTM隐藏层大小 (默认200)')
    parser.add_argument('--dropout_lstm', type=float, default=0.2,
                        help='LSTM dropout (默认0.2)')
    parser.add_argument('--linear_hidden_size', type=int, default=200,
                        help='线性层隐藏大小 (默认200)')
    parser.add_argument('--train', '-t', type=int, default=10,
                        help='训练轮数 (默认10)')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                        help='批次大小 (默认16)')

    args = parser.parse_args()

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
    )

    print("\n模型创建完成，开始训练...")
    print(f"  note_embedding_dim: {args.note_embedding_dim}")
    print(f"  meta_embedding_dim: {args.meta_embedding_dim}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  lstm_hidden_size: {args.lstm_hidden_size}")
    print(f"  训练轮数: {args.train}")
    print(f"  批次大小: {args.batch_size}")

    # 训练
    deepbach.train(
        main_voice_index=0,
        batch_size=args.batch_size,
        num_epochs=args.train
    )

    print("\n训练完成!")


if __name__ == '__main__':
    main()
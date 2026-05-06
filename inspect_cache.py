"""
查看已缓存的数据集文件内容
"""
import os
import torch

# 切换到code目录
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.chorale_dataset import ChoraleBeatsDataset, ChoraleDataset

def main():
    dm = DatasetManager()
    cache_dir = dm.cache_dir

    print("=" * 60)
    print("DeepBach 数据缓存查看器")
    print("=" * 60)
    print(f"\n缓存目录: {cache_dir}\n")

    # 查看 datasets 目录
    datasets_dir = os.path.join(cache_dir, 'datasets')
    print("=" * 60)
    print("1. Dataset 文件 (数据集参数)")
    print("=" * 60)
    if os.path.exists(datasets_dir):
        for f in os.listdir(datasets_dir):
            fp = os.path.join(datasets_dir, f)
            if '_test' in fp:
                continue  # 跳过测试数据集文件
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"\n文件: {f}")
            print(f"大小: {size_mb:.2f} MB")

            # 加载并显示内容
            try:
                dataset = torch.load(fp, map_location='cpu', weights_only=False)
                print(f"类型: {type(dataset).__name__}")
                print(f"\n内容预览:")
                print(f"  - voice_ids: {dataset.voice_ids}")
                print(f"  - name: {dataset.name}")
                print(f"  - sequences_size: {dataset.sequences_size}")
                print(f"  - subdivision: {dataset.subdivision}")
                print(f"  - num_voices: {dataset.num_voices}")
                print(f"  - metadatas: {[m.name for m in dataset.metadatas]}")
                print(f"  - voice_ranges: {dataset.voice_ranges}")

                # 显示index字典的大小
                if dataset.index2note_dicts:
                    print(f"\n  音高索引表 (note2index):")
                    for i, d in enumerate(dataset.note2index_dicts):
                        print(f"    声部{i}: {len(d)} 个音高条目")
                        # if i == 0:  # 只显示第一个声部的样例
                        samples = list(d.items())[:5]
                        print(f"      样例: {samples}")

                print(f"\n  缓存的tensor_dataset是否加载: {dataset._tensor_dataset is not None}")
            except Exception as e:
                print(f"  加载失败: {e}")
    else:
        print(f"目录不存在: {datasets_dir}")

    # 查看 tensor_datasets 目录
    tensor_dir = os.path.join(cache_dir, 'tensor_datasets')
    print("\n" + "=" * 60)
    print("2. TensorDataset 文件 (训练数据)")
    print("=" * 60)
    if os.path.exists(tensor_dir):
        for f in os.listdir(tensor_dir):
            fp = os.path.join(tensor_dir, f)
            if '_test' in fp:
                continue  # 跳过测试数据集文件
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"\n文件: {f}")
            print(f"大小: {size_mb:.2f} MB")

            try:
                tensor_dataset = torch.load(fp, map_location='cpu', weights_only=False)
                print(f"类型: {type(tensor_dataset).__name__}")
                print(f"长度: {len(tensor_dataset)} 个样本")

                # 获取张量形状
                if len(tensor_dataset) > 0:
                    tensors = tensor_dataset[:]
                    print(f"\n  张量形状:")
                    for i, t in enumerate(tensors):
                        print(f"    tensor[{i}].shape: {t.shape}")

                    # 显示具体数据
                    print(f"\n  数据内容预览 (第0个样本):")
                    chorale = tensors[0]  # 第一个张量是chorale
                    metadata = tensors[1]  # 第二个张量是metadata

                    print(f"\n    chorale_tensor (音高索引):")
                    print(f"      shape: {chorale.shape}")
                    print(f"      声部0(女高)前10个tick: {chorale[0, :10].tolist()}")
                    print(f"      声部1(女低)前10个tick: {chorale[1, :10].tolist()}")
                    print(f"      声部2(男高)前10个tick: {chorale[2, :10].tolist()}")
                    print(f"      声部3(男低)前10个tick: {chorale[3, :10].tolist()}")

                    print(f"\n    metadata_tensor (元数据):")
                    print(f"      shape: {metadata.shape}")
                    print(f"      声部0前5个tick的元数据:")
                    for tick in range(5):
                        print(f"        tick {tick}: {metadata[0, 0, tick].tolist()}")

                    # 显示原始音符名
                    print(f"\n    将音高索引转回音符名 (声部0前10个tick):")
                    # 需要重新加载index2note_dicts来转换
                    dataset_file = os.path.join(datasets_dir, os.listdir(datasets_dir)[0] if os.listdir(datasets_dir) else None)
                    if dataset_file:
                        ds = torch.load(dataset_file, map_location='cpu', weights_only=False)
                        for tick in range(10):
                            idx = chorale[0, 0, tick].item()
                            note_name = ds.index2note_dicts[0].get(idx, '?')
                            print(f"        tick {tick}: index={idx} -> {note_name}")

            except Exception as e:
                print(f"  加载失败: {e}")
    else:
        print(f"目录不存在: {tensor_dir}")

    # 总计
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    total_size = 0
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
    print(f"缓存目录总大小: {total_size / (1024*1024):.2f} MB")
    print(f"缓存目录: {cache_dir}")


if __name__ == '__main__':
    main()
"""
Standalone script to preprocess chorale data locally.
Outputs cache files and tells you where they are.
"""
import os
import shutil

# Change to code directory
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.metadata import TickMetadata, FermataMetadata, KeyMetadata


def main():
    print("=" * 60)
    print("DeepBach 数据预处理脚本")
    print("=" * 60)

    # Create dataset manager
    dm = DatasetManager()

    # Configuration
    subdivision = 4
    metadatas = [
        TickMetadata(subdivision=subdivision),
        FermataMetadata(),
        KeyMetadata()
    ]

    print(f"\n配置:")
    print(f"  细分: {subdivision} (每beat的tick数)")
    print(f"  声部: [0, 1, 2, 3] (女高、女低、男高、男低)")
    print(f"  元数据: {[m.name for m in metadatas]}")

    # Get/create dataset (using ChoraleBeatsDataset for faster processing)
    print("\n开始处理数据...")
    dataset = dm.get_dataset(
        name='bach_chorales',  # Use test set (only 10 chorales for quick testing)
        voice_ids=[0, 1, 2, 3],
        metadatas=metadatas,
        sequences_size=8,
        subdivision=subdivision
    )

    # Get cache directory paths
    cache_dir = dm.cache_dir
    tensor_cache = os.path.join(cache_dir, 'tensor_datasets')
    dataset_cache = os.path.join(cache_dir, 'datasets')

    print("\n" + "=" * 60)
    print("数据处理完成!")
    print("=" * 60)

    # List output files
    print("\n生成的缓存文件:")

    # Tensor dataset files
    print(f"\n1. TensorDataset 目录: {tensor_cache}")
    if os.path.exists(tensor_cache):
        for f in os.listdir(tensor_cache):
            fp = os.path.join(tensor_cache, f)
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"   - {f} ({size_mb:.2f} MB)")

    # Dataset files
    print(f"\n2. Dataset 目录: {dataset_cache}")
    if os.path.exists(dataset_cache):
        for f in os.listdir(dataset_cache):
            fp = os.path.join(dataset_cache, f)
            size_mb = os.path.getsize(fp) / (1024 * 1024)
            print(f"   - {f} ({size_mb:.2f} MB)")

    # Summary
    print("\n" + "=" * 60)
    print("用于训练的所有文件:")
    print(f"  {dataset_cache}")
    print("=" * 60)

    # Instructions for cloud server
    print("""
云服务器部署说明:
================

1. 将本地缓存目录复制到云服务器:
   rsync -avz --progress {cache_dir}/ root@your-cloud:/path/to/DeepBach/code/DatasetManager/

2. 或者打包后上传:
   cd {parent_dir}
   tar -czvf deepbach_cache.tar.gz dataset_cache/
   scp deepbach_cache.tar.gz root@your-cloud:/path/to/DeepBach/code/DatasetManager/
   ssh root@your-cloud "cd /path/to/DeepBach/code/DatasetManager && tar -xzvf deepbach_cache.tar.gz"

3. 云服务器上验证文件存在:
   ls -la DatasetManager/dataset_cache/datasets/
   ls -la DatasetManager/dataset_cache/tensor_datasets/
""".format(cache_dir=cache_dir, parent_dir=os.path.dirname(cache_dir)))


if __name__ == '__main__':
    main()
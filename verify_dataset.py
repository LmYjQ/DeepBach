"""
验证数据集处理是否正确
流程：
1. 从 music21 加载原始赞美诗
2. 用 DatasetManager 加载缓存的数据集
3. 取出一两个样本，手动用 music21 处理并对比
"""
import torch
import music21
import numpy as np

from DatasetManager.chorale_dataset import ChoraleBeatsDataset
from DatasetManager.metadata import TickMetadata, FermataMetadata, KeyMetadata
from DatasetManager.dataset_manager import DatasetManager


def load_original_chorale(chorale_id=0):
    """从 music21 corpus 加载原始赞美诗"""
    chorales = list(music21.corpus.chorales.Iterator())
    chorale = chorales[chorale_id]
    # 获取赞美诗的名称/元数据
    chorale.metadata = music21.metadata.Metadata(corale_id=chorale_id)
    chorale.metadata.title = chorales[chorale_id].metadata.title if chorales[chorale_id].metadata else f"Chorale {chorale_id}"
    chorale.metadata.composer = chorales[chorale_id].metadata.composer if chorales[chorale_id].metadata else "Bach"
    return chorale, chorales[chorale_id].metadata


def print_score_info(chorale, metadata):
    """打印乐谱详细信息"""
    print("\n" + "=" * 60)
    print(f"乐谱详情: {metadata.title if metadata else 'Unknown'}")
    print("=" * 60)
    print(f"作曲家: {metadata.composer if metadata else 'Bach'}")

    # 显示乐谱的五线谱布局
    print("\n[乐谱预览 - 每个声部的第一个音符]:")
    for i, part in enumerate(chorale.parts):
        part_name = part.partName if part.partName else f"Voice {i}"
        first_note = None
        for elem in part.flat.notesAndRests:
            if hasattr(elem, 'isNote') and elem.isNote:
                first_note = elem
                break
        if first_note:
            print(f"  {part_name}: {first_note.nameWithOctave} (pitch={first_note.pitch.midi})")
        else:
            print(f"  {part_name}: (无音符)")

    # 显示乐谱前几个小节
    print(f"\n[乐谱前8拍详细 - offset 0 到 8]:")
    for i, part in enumerate(chorale.parts):
        part_name = part.partName if part.partName else f"Voice {i}"
        print(f"\n  {part_name}:")

        notes_in_range = []
        for elem in part.flat.getElementsByOffset(0, 8):
            if elem.isNote:
                notes_in_range.append(f"{elem.nameWithOctave}({elem.offset})")
            elif elem.isRest:
                notes_in_range.append(f"Rest({elem.offset})")
            elif hasattr(elem, 'isChord') and elem.isChord:
                notes_in_range.append(f"Chord({elem.offset})")

        # 只显示前20个音符
        if len(notes_in_range) > 20:
            print(f"    {', '.join(notes_in_range[:20])} ...")
        else:
            print(f"    {', '.join(notes_in_range)}")


def verify_dataset_item(chorale_id=0, offsetStart=0, semi_tone=0):
    """
    对比原始 music21 数据和处理后的数据集

    步骤：
    1. 加载原始赞美诗
    2. 按照与 make_tensor_dataset 相同的流程处理
    3. 从数据集取出对应样本对比
    """
    print("=" * 60)
    print(f"验证 chorale_id={chorale_id}, offsetStart={offsetStart}, semi_tone={semi_tone}")
    print("=" * 60)

    # 1. 加载原始赞美诗
    chorale, metadata = load_original_chorale(chorale_id)
    print(f"\n[1] 原始赞美诗信息:")
    print(f"    编号: {chorale_id}")
    print(f"    标题: {metadata.title}")
    print(f"    作曲家: {metadata.composer}")
    print(f"    持续时间: {chorale.duration.quarterLength} 拍")
    print(f"    lowestOffset: {chorale.flat.lowestOffset}")
    print(f"    highestOffset: {chorale.flat.highestOffset}")
    print(f"    声部数: {len(chorale.parts)}")
    for i, part in enumerate(chorale.parts):
        part_name = part.partName if part.partName else f"Voice {i}"
        print(f"    声部 {i}: {part_name}")

    # 打印乐谱详情
    # print_score_info(chorale, metadata)

    # 2. 设置参数
    subdivision = 4
    sequences_size = 8
    one_beat = 1.0
    start_tick = int(offsetStart * subdivision)
    end_tick = int((offsetStart + sequences_size) * subdivision)

    # 3. 创建临时 dataset 对象来获取方法
    from DatasetManager.helpers import standard_name, standard_note, SLUR_SYMBOL, START_SYMBOL, END_SYMBOL

    # 获取 note2index dicts 和 voice_ranges（从缓存的数据集）
    dataset_manager = DatasetManager()
    metadatas = [
        TickMetadata(subdivision=4),
        FermataMetadata(),
        KeyMetadata()
    ]
    dataset = dataset_manager.get_dataset(
        name='bach_chorales',
        voice_ids=[0, 1, 2, 3],
        metadatas=metadatas,
        sequences_size=8,
        subdivision=4
    )

    # 4. 获取原始赞美诗的转调张量
    interval_type, interval_nature = music21.interval.convertSemitoneToSpecifierGeneric(semi_tone)
    transposition_interval = music21.interval.Interval(str(interval_nature) + str(interval_type))
    chorale_transposed = chorale.transpose(transposition_interval)

    print(f"\n[2] 转调后赞美诗 (semi_tone={semi_tone}):")
    print(f"    原调: {music21.analysis.floatingKey.KeyAnalyzer(chorale).run()[0].name}")
    print(f"    转调后: {music21.analysis.floatingKey.KeyAnalyzer(chorale_transposed).run()[0].name}")

    # 5. 获取 tensor（用 dataset 的方法）
    chorale_tensor = dataset.get_score_tensor(
        chorale_transposed,
        offsetStart=0.,
        offsetEnd=chorale_transposed.flat.highestTime
    )
    metadata_tensor = dataset.get_metadata_tensor(chorale_transposed)

    print(f"\n[3] 完整张量形状:")
    print(f"    chorale_tensor: {chorale_tensor.shape}")
    print(f"    metadata_tensor: {metadata_tensor.shape}")

    # 6. 提取窗口
    local_chorale = dataset.extract_score_tensor_with_padding(
        chorale_tensor, start_tick, end_tick)
    local_metadata = dataset.extract_metadata_with_padding(
        metadata_tensor, start_tick, end_tick)

    print(f"\n[4] 提取窗口 (start_tick={start_tick}, end_tick={end_tick}):")
    print(f"    local_chorale: {local_chorale.shape}")
    print(f"    local_metadata: {local_metadata.shape}")

    # 7. 从数据集获取对应样本对比
    print(f"\n[5] 与数据集对比:")
    tensor_dataset = dataset.tensor_dataset
    # 找到对应样本（通过遍历找第一个匹配的）
    found = False
    for i in range(min(1000, len(tensor_dataset))):  # 只检查前1000个
        chorale_item, metadata_item = tensor_dataset[i]
        # 检查是否匹配
        if (chorale_item.shape == local_chorale.shape and
            torch.allclose(chorale_item.long(), local_chorale.long())):
            print(f"    样本 {i} 匹配!")
            print(f"    数据集样本形状: chorale={chorale_item.shape}, metadata={metadata_item.shape}")
            found = True
            break

    if not found:
        print("    未在检查范围内找到匹配样本")
        print(f"\n    手动处理的样本前5个值:")
        print(f"    chorale[0, :5]: {local_chorale[0, :5]}")
        print(f"    chorale[1, :5]: {local_chorale[1, :5]}")
        print(f"    metadata[0, :5, 0]: {local_metadata[0, :5, 0]}")  # tick metadata

    # 8. 手动验证第一个样本
    print(f"\n[6] 数据集第一个样本信息:")
    chorale_item, metadata_item = tensor_dataset[0]
    print(f"    chorale_item.shape: {chorale_item.shape}")
    print(f"    metadata_item.shape: {metadata_item.shape}")
    print(f"    chorale_item[0, :8]: {chorale_item[0, :8]}")
    print(f"    metadata_item[0, :8, 0]: {metadata_item[0, :8, 0]}")  # tick

    return local_chorale, local_metadata, dataset


def index_to_note_name(index2note_dict):
    """把索引转回音符名字"""
    return {v: k for k, v in index2note_dict.items()}


def load_chorale_from_cache(dataset, chorale_id=0):
    """
    从缓存的数据集中解析第n首曲子的所有内容
    由于数据集是按 (offset, transposition) 采样的，需要找出属于同一首曲子的样本
    """
    from DatasetManager.helpers import standard_name, standard_note, SLUR_SYMBOL, START_SYMBOL, END_SYMBOL

    tensor_dataset = dataset.tensor_dataset

    print("=" * 60)
    print(f"从数据集解析第 {chorale_id} 首曲子")
    print("=" * 60)

    # 获取 chorale_id 对应的样本范围
    # 遍历数据集，记录每首曲子的样本范围
    chorale_ranges = {}
    current_chorale_id = 0
    start_idx = 0

    for i in range(len(tensor_dataset)):
        chorale_item, metadata_item = tensor_dataset[i]

        # 从metadata中尝试推断是哪首曲子
        # 由于数据集中没有直接存储chorale_id，我们需要根据数据结构推断

        if i % 10000 == 0 and i > 0:
            print(f"  进度: {i}/{len(tensor_dataset)}")

        # 简单方法：假设每首曲子约产生600-800个样本
        # 351首曲子，267031样本，约760样本/首
        samples_per_chorale = len(tensor_dataset) // 351

        if i > 0 and i % samples_per_chorale == 0:
            chorale_ranges[current_chorale_id] = (start_idx, i)
            current_chorale_id += 1
            start_idx = i

    chorale_ranges[current_chorale_id] = (start_idx, len(tensor_dataset))

    if chorale_id not in chorale_ranges:
        print(f"  错误: 没有找到第 {chorale_id} 首曲子")
        return None

    start_idx, end_idx = chorale_ranges[chorale_id]
    print(f"\n第 {chorale_id} 首曲子包含样本索引: {start_idx} ~ {end_idx} (共 {end_idx - start_idx} 个样本)")

    # 获取这首歌的所有样本
    print(f"\n解析样本内容...")
    all_voices_notes = [[] for _ in range(4)]
    all_metadata = [[] for _ in range(4)]

    # 采样显示：每隔一定步长取一个样本
    step = max(1, (end_idx - start_idx) // 20)
    for sample_idx in range(start_idx, end_idx, step):
        chorale_item, metadata_item = tensor_dataset[sample_idx]

        for voice_id in range(4):
            # 获取该声部的音符序列（前32个tick）
            note_indexes = chorale_item[voice_id, :].numpy()

            # 转换为音符名字
            note_names = []
            for idx in note_indexes:
                name = dataset.index2note_dicts[voice_id].get(idx, '?')
                if name == SLUR_SYMBOL:
                    note_names.append('_')  # 用下划线表示延音符
                elif name == START_SYMBOL:
                    note_names.append('S')
                elif name == END_SYMBOL:
                    note_names.append('E')
                else:
                    note_names.append(name)

            all_voices_notes[voice_id].append(note_names)

            # metadata
            meta = metadata_item[voice_id, :, :].numpy()
            all_metadata[voice_id].append(meta)

    return all_voices_notes, all_metadata, chorale_ranges


def describe_sample(chorale_tensor, metadata_tensor, dataset, sample_idx=0):
    """描述一个样本的内容"""
    print("\n" + "=" * 60)
    print(f"描述样本 {sample_idx}")
    print("=" * 60)

    chorale_item, metadata_item = dataset.tensor_dataset[sample_idx]

    print(f"\n形状:")
    print(f"  chorale: {chorale_item.shape}")
    print(f"  metadata: {metadata_item.shape}")

    print(f"\n每声部前10个音符索引:")
    for voice_id in range(4):
        indexes = chorale_item[voice_id, :10].numpy()
        names = []
        for idx in indexes:
            note_name = dataset.index2note_dicts[voice_id].get(idx, '?')
            names.append(note_name)
        print(f"  声部 {voice_id}: {list(zip(indexes.tolist(), names))}")

    print(f"\nMetadata tick (前10个tick):")
    for voice_id in range(4):
        ticks = metadata_item[voice_id, :10, 0].numpy()
        print(f"  声部 {voice_id}: {ticks.tolist()}")


def find_offset_zero_samples(dataset, max_samples=20):
    """
    在数据集中查找 offset=0 位置的样本
    offset=0 意味着第一个实际的音符（不是 START_SYMBOL）应该从 tick 0 开始
    因为 START_SYMBOL 用于填充负偏移的情况
    """
    from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL

    tensor_dataset = dataset.tensor_dataset
    subdivision = dataset.subdivision

    print("\n" + "=" * 60)
    print("查找 offset=0 附近的样本")
    print("=" * 60)

    found_samples = []

    for i in range(len(tensor_dataset)):
        chorale_item, metadata_item = tensor_dataset[i]

        # 找到第一个非 START_SYMBOL 的位置
        for voice_id in range(4):
            first_note_idx = None
            first_note_tick = None

            for tick_pos in range(chorale_item.shape[1]):
                note_idx = chorale_item[voice_id, tick_pos].item()
                note_name = dataset.index2note_dicts[voice_id].get(note_idx, '?')
                if note_name != START_SYMBOL:
                    first_note_idx = note_idx
                    first_note_tick = tick_pos
                    break

            # 检查是否 offset=0 (第一个实际音符在 tick 0 或很早的位置)
            if first_note_tick is not None and first_note_tick <= 2:
                found_samples.append((i, chorale_item.clone(), metadata_item.clone()))
                break

        if len(found_samples) >= max_samples:
            break

    return found_samples


def describe_specific_sample(dataset, sample_idx=0):
    """描述特定索引的样本内容"""
    from DatasetManager.helpers import SLUR_SYMBOL, START_SYMBOL, END_SYMBOL

    print("\n" + "=" * 60)
    print(f"描述样本 {sample_idx}")
    print("=" * 60)

    chorale_item, metadata_item = dataset.tensor_dataset[sample_idx]

    print(f"\n形状:")
    print(f"  chorale: {chorale_item.shape}")
    print(f"  metadata: {metadata_item.shape}")

    voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']

    print(f"\n每声部音符序列 (全部32个tick):")
    for voice_id in range(4):
        indexes = chorale_item[voice_id, :].numpy()
        note_names = []
        for idx in indexes:
            name = dataset.index2note_dicts[voice_id].get(idx, '?')
            if name == SLUR_SYMBOL:
                note_names.append('_')
            elif name == START_SYMBOL:
                note_names.append('S')
            elif name == END_SYMBOL:
                note_names.append('E')
            else:
                note_names.append(name)

        # 找到第一个非 S 的位置
        first_real_note = 0
        for j, n in enumerate(note_names):
            if n != 'S':
                first_real_note = j
                break

        print(f"\n  {voice_names[voice_id]}:")
        print(f"    全部: {' '.join(note_names)}")
        print(f"    前32: {' '.join(note_names[:32])}")
        print(f"    首个音符位置: tick {first_real_note} = beat {first_real_note / 4:.2f}")

        ticks = metadata_item[voice_id, :, 0].numpy()
        keys = metadata_item[voice_id, :, 2].numpy()
        print(f"    tick:  {ticks.tolist()[:32]}")
        print(f"    key:   {keys.tolist()[:32]}")


if __name__ == '__main__':
    print("从数据集解析曲子内容\n")

    # 加载数据集
    dataset_manager = DatasetManager()
    metadatas = [
        TickMetadata(subdivision=4),
        FermataMetadata(),
        KeyMetadata()
    ]
    dataset = dataset_manager.get_dataset(
        name='bach_chorales',
        voice_ids=[0, 1, 2, 3],
        metadatas=metadatas,
        sequences_size=8,
        subdivision=4
    )

    # 描述第一个样本
    print("第一个样本 (sample_idx=0):")
    describe_specific_sample(dataset, sample_idx=0)

    # 查找 offset=0 附近的样本
    print("\n\n" + "=" * 60)
    print("在数据集中搜索 offset=0 的样本...")
    print("=" * 60)

    found_samples = find_offset_zero_samples(dataset, max_samples=5)
    print(f"\n找到 {len(found_samples)} 个 offset=0 的样本")

    for sample_idx, chorale_item, metadata_item in found_samples:
        print(f"\n--- 样本 {sample_idx} ---")
        describe_specific_sample(dataset, sample_idx=sample_idx)

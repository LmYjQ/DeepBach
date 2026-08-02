# 简谱JSON预处理文档

## 概述

将自定义简谱JSON格式转换为DeepBach模型所需的TensorDataset格式。

## 输入格式 (简谱JSON)

```json
{
  "title": "歌曲名",
  "tempo": 60,
  "beatsPerBar": 4,
  "notes": [
    {
      "value": 1-7,     // 音高: 1=do, 2=re, 3=mi, 4=fa, 5=sol, 6=la, 7=si, 0=休止符
      "octave": 0,      // 八度偏移 (0=中央八度, 1=高八度, -1=低八度)
      "duration": 1.0   // 时值 (1=一拍, 0.5=半拍, 0.125=32分音符等)
    }
  ]
}
```

**注意**: `value: "bar"` 表示小节线，会被自动过滤。

## 输出格式

```
chorale_tensor: (N, 1, sequence_size * subdivision)
  - N: 样本数量
  - 1: 声部数量 (单声部)
  - sequence_size * subdivision: 每样本tick数

metadata_tensor: (N, sequence_size * subdivision, 2)
  - 每行 [tick_position, voice_id]
  - tick_position: 0~subdivision-1，表示在拍内的位置
  - voice_id: 声部ID (固定为0)
```

## 核心映射

### 简谱到MIDI

| value | 音名 | MIDI (octave=0) |
|-------|------|-----------------|
| 1 | do | 60 (C4) |
| 2 | re | 62 (D4) |
| 3 | mi | 64 (E4) |
| 4 | fa | 65 (F4) |
| 5 | sol | 67 (G4) |
| 6 | la | 69 (A4) |
| 7 | si | 71 (B4) |

八度偏移: `midi_pitch = base_midi + (octave * 12)`

### 特殊符号

| 符号 | 含义 | 说明 |
|------|------|------|
| `rest` | 休止符 | value=0 时使用 |
| `slur` | 延长记号 | 音符延续 (非起点) |
| `START` | 起始符号 | DeepBach模型用 |
| `END` | 结束符号 | DeepBach模型用 |

## 使用方法

### 预处理

```bash
cd D:\code\music\deepbach_learn\code
python preprocess_chinese_score.py --input <json_path> --output <output_dir>
```

参数:
- `--input, -i`: 输入JSON文件路径
- `--output, -o`: 输出目录 (默认: ./preprocessed_data)
- `--sequence_size, -s`: 序列长度(拍) (默认: 8)
- `--subdivision, -d`: 每拍tick数 (默认: 8，即32分音符精度)

### 验证

```bash
python preprocess_chinese_score.py --verify --input <json_path>
```

会输出前20个样本的详细信息用于对比原始数据。

## 预处理流程

1. **加载JSON** - 读取文件，过滤掉 `value: "bar"` 和 `value: null` 的条目
2. **转MIDI序列** - 将简谱转为MIDI音高序列，记录每个音符的起点/延续标记
3. **创建映射表** - 建立 note2index 和 index2note 映射
4. **滑动窗口采样** - 步长为1拍，生成训练样本
5. **保存** - 输出为 `.pt` 文件

## 时值精度

`subdivision=8` 时可表示 0.125 拍的精度:

| duration | subdivision=4 | subdivision=8 |
|----------|--------------|--------------|
| 1.0 | 4 ticks | 8 ticks |
| 0.5 | 2 ticks | 4 ticks |
| 0.25 | 1 tick | 2 ticks |
| 0.125 | **0 ticks (丢失!)** | 1 tick |

建议使用 `subdivision=8` 以支持更精细的时值。

## 示例

输入JSON:
```json
{"value": 5, "octave": 0, "duration": 1.0}
```

处理后 (subdivision=4):
- MIDI: 67
- 索引序列: `[67, slur, slur, slur]` (4个tick)

输出tensor (8拍序列):
```
样本0: 67,slur,slur,slur, 60,slur,slur,slur, ...
原始:  5_0,_,_,_,     1_0,_,_,_,    ...
```



# 1. 云庆5.9.json
python D:\code\music\deepbach_learn\code\json_to_midi.py -i "D:\code\music\qmx_reader\dataset_da\云庆5.9.json" -o "D:\code\music\qmx_reader\dataset_midi\云庆5.9.mid"

# 2. 四合如意.json
python D:\code\music\deepbach_learn\code\json_to_midi.py -i "D:\code\music\qmx_reader\dataset_da\四合如意.json" -o "D:\code\music\qmx_reader\dataset_midi\四合如意.mid"

# 3. 慢三六.json
python D:\code\music\deepbach_learn\code\json_to_midi.py -i "D:\code\music\qmx_reader\dataset_da\慢三六.json" -o "D:\code\music\qmx_reader\dataset_midi\慢三六.mid"

# 4. 行街（10分钟）全.json
python D:\code\music\deepbach_learn\code\json_to_midi.py -i "D:\code\music\qmx_reader\dataset_da\行街（10分钟）全.json" -o "D:\code\music\qmx_reader\dataset_midi\行街（10分钟）全.mid"

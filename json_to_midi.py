"""
将简谱JSON格式转换为MIDI文件

简谱JSON格式:
{
  "title": "歌曲名",
  "tempo": 60,          # BPM
  "beatsPerBar": 4,     # 每小节拍数
  "notes": [
    {"value": 1-7, "octave": 0, "duration": 1.0}  # value: 1=do, 2=re, ..., 7=si, 0=休止
  ]
}

Usage:
  python json_to_midi.py --input <json_path> --output <output.mid>
  python json_to_midi.py --input <json_path> --output <output.mid> --tempo 90
"""
import json
import argparse
import os

from music21 import stream, note, tempo, meter

# 简谱音名到 MIDI 的映射 (首调，中央C为基准)
SOLFEGE_TO_MIDI_BASE = {
    1: 60,   # do -> C4
    2: 62,   # re -> D4
    3: 64,   # mi -> E4
    4: 65,   # fa -> F4
    5: 67,   # sol -> G4
    6: 69,   # la -> A4
    7: 71,   # si -> B4
}


def json_to_music21(json_path):
    """
    将简谱JSON转换为music21 Stream

    Args:
        json_path: 简谱JSON文件路径

    Returns:
        music21.stream.Score
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    title = data.get('title', 'Unknown')
    tempo_bpm = data.get('tempo', 60)
    beats_per_bar = data.get('beatsPerBar', 4)
    notes = data.get('notes', [])

    # 过滤非音符数据
    notes = [n for n in notes if str(n.get('value', '')).lower() not in ['bar', 'space']]
    notes = [n for n in notes if n.get('value') is not None]

    # 创建stream
    score = stream.Stream()
    score.insert(0, tempo.MetronomeMark(tempo_bpm))
    score.insert(0, meter.TimeSignature(f'{beats_per_bar}/4'))

    notes_stream = stream.Stream()
    for note_data in notes:
        value = int(note_data['value'])
        octave = note_data.get('octave', 0)
        duration = note_data.get('duration', 1.0)

        if value == 0:
            # 休止符
            n = note.Rest()
        else:
            # 计算MIDI音高
            base_midi = SOLFEGE_TO_MIDI_BASE[value]
            midi_pitch = base_midi + (octave * 12)
            n = note.Note(midi_pitch)

        # 设置时值 (music21: 1 beat = 1 quarter note)
        n.quarterLength = duration
        notes_stream.append(n)

    score.insert(0, notes_stream)
    return score


def json_to_midi(json_path, output_path, tempo=None):
    """
    将简谱JSON直接转换为MIDI文件

    Args:
        json_path: 输入JSON文件路径
        output_path: 输出MIDI文件路径
        tempo: 可选，覆盖JSON中的tempo值
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if tempo is None:
        tempo = data.get('tempo', 60)

    score = json_to_music21(json_path)

    # 更新tempo
    if tempo != data.get('tempo', 60):
        for part in score.parts:
            for elem in part:
                if isinstance(elem, tempo.MetronomeMark):
                    elem.number = tempo
                    break

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    score.write('midi', fp=output_path)
    print(f'MIDI已保存: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='简谱JSON转MIDI')
    parser.add_argument('--input', '-i', required=True, help='输入JSON文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出MIDI文件路径')
    parser.add_argument('--tempo', '-t', type=int, default=None, help='Tempo (BPM, 覆盖JSON值)')

    args = parser.parse_args()
    json_to_midi(args.input, args.output, args.tempo)
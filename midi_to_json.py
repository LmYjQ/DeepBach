"""
将MIDI文件转换回简谱JSON格式

输入: MIDI文件
输出: JSON文件，包含:
  - notes: 音符数组，每个元素有 value, octave, duration
  - time_signature 事件: {"value": "3/4", "type": "time_signature"}
  - tempo 事件: {"value": 60, "type": "tempo"}

Usage:
  python midi_to_json.py -i <midi_path> -o <output.json>
"""
import json
import argparse

from music21 import midi, meter, tempo


SOLFEGE_TO_MIDI_BASE = {
    1: 60, 2: 62, 3: 64, 4: 65, 5: 67, 6: 69, 7: 71
}


def midi_pitch_to_solfege(midi_pitch):
    """将MIDI音高转换为简谱格式 (value, octave)"""
    if midi_pitch == 0:
        return 0, 0

    rel = midi_pitch - 60
    natural_notes = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}
    note_idx = rel % 12
    octave_offset = rel // 12

    if note_idx in natural_notes:
        return natural_notes[note_idx], octave_offset

    # 找到最近的自然音
    lower_idx = max([k for k in natural_notes if k <= note_idx])
    upper_idx = min([k for k in natural_notes if k >= note_idx])
    if note_idx - lower_idx <= upper_idx - note_idx:
        value = natural_notes[lower_idx]
    else:
        value = natural_notes[upper_idx]
    return value, octave_offset


def midi_to_json(midi_path, output_path):
    """
    将MIDI文件转换为简谱JSON

    Args:
        midi_path: 输入MIDI文件路径
        output_path: 输出JSON文件路径
    """
    score = midi.translate.midiFilePathToStream(midi_path)

    # 收集事件
    notes_list = []
    current_offset = 0.0

    for elem in score.flatten():
        # 获取元素在总音符列表中的位置(偏移量)
        elem_offset = elem.offset if hasattr(elem, 'offset') else current_offset

        # tempo 事件
        if isinstance(elem, tempo.MetronomeMark):
            notes_list.append({
                'value': elem.number,
                'type': 'tempo',
                'offset': elem_offset
            })

        # time signature 事件
        elif isinstance(elem, meter.TimeSignature):
            ts_str = f'{elem.beatCount}/{int(elem.beatDuration.quarterLength * 4)}'
            notes_list.append({
                'value': ts_str,
                'type': 'time_signature',
                'offset': elem_offset
            })

        # 音符/休止符
        elif hasattr(elem, 'pitch') and elem.pitch is not None:
            from music21 import note
            if isinstance(elem, note.Note):
                value, octave = midi_pitch_to_solfege(elem.pitch.midi)
                notes_list.append({
                    'value': value,
                    'octave': octave,
                    'duration': float(elem.quarterLength),
                    'offset': elem_offset
                })
            elif isinstance(elem, note.Rest):
                notes_list.append({
                    'value': 0,
                    'octave': 0,
                    'duration': float(elem.quarterLength),
                    'offset': elem_offset
                })

    # 整理输出
    output = {
        'title': score.metadata.title if score.metadata and score.metadata.title else 'Unknown',
        'tempo': 60,
        'beatsPerBar': 4,
        'notes': []
    }

    # 提取tempo和time_signature
    for item in notes_list:
        if item.get('type') == 'tempo':
            output['tempo'] = item['value']
        elif item.get('type') == 'time_signature':
            ts_parts = item['value'].split('/')
            output['beatsPerBar'] = int(ts_parts[0])
            # 作为普通音符添加
            output['notes'].append({
                'value': item['value'],
                'type': 'time_signature'
            })
        else:
            output['notes'].append({
                'value': item['value'],
                'octave': item['octave'],
                'duration': item['duration']
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'已保存: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIDI转简谱JSON')
    parser.add_argument('--input', '-i', required=True, help='输入MIDI文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出JSON文件路径')

    args = parser.parse_args()
    midi_to_json(args.input, args.output)
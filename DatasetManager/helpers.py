import music21
from itertools import islice

from music21 import note, harmony, expressions

# constants
SLUR_SYMBOL = '__'
START_SYMBOL = 'START'
END_SYMBOL = 'END'
REST_SYMBOL = 'rest'
OUT_OF_RANGE = 'OOR'
PAD_SYMBOL = 'XX'


def standard_name(note_or_rest, voice_range=None):
    """
    Convert music21 objects to str
    :param note_or_rest:
    :return:
    """
    if isinstance(note_or_rest, note.Note):
        if voice_range is not None:
            min_pitch, max_pitch = voice_range
            pitch = note_or_rest.pitch.midi
            if pitch < min_pitch or pitch > max_pitch:
                return OUT_OF_RANGE
        return note_or_rest.nameWithOctave
    if isinstance(note_or_rest, note.Rest):
        return note_or_rest.name  # == 'rest' := REST_SYMBOL
    if isinstance(note_or_rest, str):
        return note_or_rest

    if isinstance(note_or_rest, harmony.ChordSymbol):
        return note_or_rest.figure
    if isinstance(note_or_rest, expressions.TextExpression):
        return note_or_rest.content


def standard_note(note_or_rest_string):
    """
    Convert str representing a music21 object to this object
    :param note_or_rest_string:
    :return:
    """
    if note_or_rest_string == 'rest':
        return note.Rest()
    # treat other additional symbols as rests
    elif (note_or_rest_string == END_SYMBOL
          or
          note_or_rest_string == START_SYMBOL
          or
          note_or_rest_string == PAD_SYMBOL):
        # print('Warning: Special symbol is used in standard_note')
        return note.Rest()
    elif note_or_rest_string == SLUR_SYMBOL:
        # print('Warning: SLUR_SYMBOL used in standard_note')
        return note.Rest()
    elif note_or_rest_string == OUT_OF_RANGE:
        # print('Warning: OUT_OF_RANGE used in standard_note')
        return note.Rest()
    else:
        return note.Note(note_or_rest_string)


class ShortChoraleIteratorGen:
    """
    Class used for debugging
    when called, it returns an iterator over 3 Bach chorales,
    similar to music21.corpus.chorales.Iterator()
    """

    def __init__(self):
        pass

    def __call__(self):
        it = (
            chorale
            for chorale in
            islice(music21.corpus.chorales.Iterator(), 3)
        )
        return it.__iter__()


# ---------- Beat-level tokenization helpers ----------

def beat_to_midi(pattern):
    """
    把一拍内编码的 pattern 字符串解码为 (midi_or_None, duration_in_beats) 列表。

    Args:
        pattern: 形如 ``"72@0.5|74@0.25|R@0.25"`` 的字符串，每个子项为
                 ``"<midi_or_R>@<duration_in_beats>"``，时长之和通常为 1。

    Returns:
        list of (midi: int | None, duration: float)
            - ``midi=None`` 表示休止符 (来自 ``R@...``)
            - ``duration`` 单位为**拍** (quarterLength)，例如 0.5 = 半拍
        若 ``pattern`` 为空/None，返回 ``[(None, 1.0)]`` 兜底。

    Examples:
        >>> beat_to_midi("72@1")
        [(72, 1.0)]
        >>> beat_to_midi("72@0.5|74@0.5")
        [(72, 0.5), (74, 0.5)]
        >>> beat_to_midi("R@0.5|62@0.5")
        [(None, 0.5), (62, 0.5)]
        >>> beat_to_midi("72@2")
        [(72, 2.0)]
    """
    if not pattern:
        return [(None, 1.0)]
    out = []
    for part in pattern.split('|'):
        part = part.strip()
        if not part:
            continue
        if '@' not in part:
            # 兜底: 整段当作 MIDI 数字，整拍
            try:
                out.append((int(part), 1.0))
            except ValueError:
                out.append((None, 1.0))
            continue
        pitch_str, dur_str = part.split('@', 1)
        try:
            duration = float(dur_str)
        except ValueError:
            duration = 1.0
        if pitch_str == 'R':
            out.append((None, duration))
        else:
            try:
                out.append((int(pitch_str), duration))
            except ValueError:
                out.append((None, duration))
    if not out:
        return [(None, 1.0)]
    return out


def is_beat_pattern(s):
    """
    粗略判断字符串是否为 beat-pattern（包含 '@' 子项分隔符）。
    用于 ``SimpleNotationDataset.tensor_to_score`` 中决定走哪个解码分支。
    """
    if not isinstance(s, str):
        return False
    return '@' in s and s not in (SLUR_SYMBOL, START_SYMBOL, END_SYMBOL,
                                   REST_SYMBOL, PAD_SYMBOL, OUT_OF_RANGE)

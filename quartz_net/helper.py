# 中文预测的label

def get_alphabet_labels():
    # 英文字母
    return ('-',
            '|',
            'e',
            't',
            'o',
            'i',
            'a',
            'n',
            's',
            'r',
            'h',
            'l',
            'd',
            'c',
            'u',
            'm',
            'p',
            'f',
            'g',
            'w',
            'y',
            'b',
            'v',
            'k',
            'x',
            'j',
            'q',
            'z')
            
def get_initial_table():
    return ['b', 'p', 'm', 'f',
            'd', 't', 'n', 'l',
            'g', 'k', 'h',
            'j', 'q', 'x',
            'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w']

def get_final_table():
    return ['i', 'u', 'v', 
            'a', 'ia', 'ua', 
            'o', 'uo', 'e', 'ie', 've',
            'ai', 'uai',
            'ei', 'uei',
            'ao', 'iao',
            'ou', 'iou',
            'an', 'ian',
            'uan', 'van',
            'en', 'in', 'uen', 'vn',
            'ang', 'iang', 'uang',
            'eng', 'ing', 'ueng',
            'ong', 'iong',
            'er', 'ê',]

def get_phoneme_labels():
    # 汉语声母韵母
    _INITIALS = get_initial_table()
    _FINALS = get_final_table()
    return ('-', '|') + tuple(
        _INITIALS + _FINALS
    )

def get_tone_labels():
    # 中文拼音音调，5是轻声
    return ('-',
            '|',
            '1',
            '2',
            '3',
            '4',
            '5')

def get_pitch_labels():
    # 歌曲音阶
    return ('-',
            '|',
            'A#3/Bb3',
            'A#4/Bb4',
            'A3',
            'A4',
            'A5',
            'B3',
            'B4',
            'C#2/Db2',
            'C#3/Db3',
            'C#4/Db4',
            'C#5/Db5',
            'C3',
            'C4',
            'C5',
            'D#3/Eb3',
            'D#4/Eb4',
            'D#5/Eb5',
            'D2',
            'D3',
            'D4',
            'D5',
            'E3',
            'E4',
            'E5',
            'F#3/Gb3',
            'F#4/Gb4',
            'F#5/Gb5',
            'F3',
            'F4',
            'F5',
            'G#3/Ab3',
            'G#4/Ab4',
            'G3',
            'G4',
            'rest')
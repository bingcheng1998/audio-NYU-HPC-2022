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

def parser_line(line):
    id, text, phoneme, note, note_duration, phoneme_duration, slur_note = line.split('|')
    phoneme = phoneme.split(' ')
    note = note.split(' ')
    note_duration = [float(i) for i in note_duration.split(' ')]
    phoneme_duration = [float(i) for i in phoneme_duration.split(' ')]
    slur_note = [int(i) for i in slur_note.split(' ')]
    assert len(phoneme) == len(note_duration) and len(phoneme_duration) == len(slur_note) and len(slur_note) == len(phoneme)
    return id, text, phoneme, note, note_duration, phoneme_duration, slur_note

def merge_note(text, phoneme, note, note_duration, slur_note):
    # 1. check whether the phoneme is in finals
    INITIALS = get_initial_table()
    FINALS = get_final_table()
    # is_final = [1 if p in FINALS else 0 for p in phoneme]
    phoneme = phoneme.copy()
    note = note.copy()
    note_duration = note_duration.copy()
    slur_note = slur_note.copy()
    j = -1
    text+='////////////////////'
    text_with_p = phoneme.copy()
    used_flag = False
    for i in range(len(text_with_p)):
        if text_with_p[i] in ['AP', 'SP']:
            continue
        if j==-1 or phoneme[i] in INITIALS or (phoneme[i-1] not in INITIALS and phoneme[i] != phoneme[i-1]):
            j+=1
            used_flag = False
        text_with_p[i] = text[j] if used_flag == False else '~'
        used_flag = True
    for i in range(len(phoneme)-1, 0, -1):
        if (note_duration[i] == note_duration[i-1] and phoneme[i-1] in INITIALS):
            del note_duration[i]
            del note[i]
            phoneme[i-1]=[phoneme[i-1],phoneme[i]]
            del phoneme[i]
            del text_with_p[i]
            del slur_note[i]
        elif phoneme[i] in FINALS or phoneme[i] in ['AP', 'SP']:
            phoneme[i] = [phoneme[i]]
    if phoneme[0] in FINALS or phoneme[0] in ['AP', 'SP']:
            phoneme[0] = [phoneme[0]]
    return text_with_p, phoneme, note, note_duration, slur_note

def get_transposed_phoneme_labels():
    phoneme_list = get_initial_table() + get_final_table()
    for x in ['iou', 'uei', 'uen', 'ueng', 'ê']:
        phoneme_list.remove(x)
    phoneme_list += ['iu', 'ui', 'un', 'AP', 'SP']
    return ['-', '|']+phoneme_list

def print_all(x): 
    for s in x:
        print(len(s), s)
from pypinyin import Style, lazy_pinyin

initial_table = ['b', 'p', 'm', 'f',
                'd', 't', 'n', 'l',
                'g', 'k', 'h',
                'j', 'q', 'x',
                'zh', 'ch', 'sh', 'r', 'z', 'c', 's']

finals_table = [['i', 'u', 'v'], # 可以与下面的配成 iao, ue
                ['a', 'o', 'e', 
                'ai', 'ei', 'ao', 'ou', 
                'an', 'en', 'on', 'in', 'vn', 'un',
                'ang', 'eng', 'ong', 'ing', 'er']]

def get_labels():
    return ['-', '|'] + initial_table + finals_table[0] + finals_table[1]

def chinese2pinyin(text):
    initials = lazy_pinyin(text, strict=True, style=Style.INITIALS, errors=lambda x: u'')
    finals = lazy_pinyin(text, strict=True, style=Style.FINALS, errors=lambda x: u'')
    pinyin = []
    for i in range(len(finals)):
        pinyin+=['-']
        if initials[i] != '':
            pinyin+=[initials[i]]
        if finals[i] != '':
            if len(finals[i])>1 and finals[i][0] in finals_table[0] and finals[i][1]!='n':
                pinyin+=[finals[i][0], finals[i][1:]]
            else: pinyin+=[finals[i]]
        if initials[i] == '' and finals[i] == '':
            pinyin+=['n']
    # if pinyin[-1] == '|':
    #     pinyin = pinyin[:-1]
    return pinyin[1:]
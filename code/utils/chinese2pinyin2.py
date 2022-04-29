from pypinyin import Style, lazy_pinyin


def get_labels():
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
            
def chinese2pinyin(text, zero_pad=False):
        pinyin = lazy_pinyin(text, strict=True,errors=lambda x: u'')
        pinyin = [i for i in '|'.join(pinyin)]
        if zero_pad:
            pinyin =['-', '-']+pinyin+['-', '-']
        return pinyin
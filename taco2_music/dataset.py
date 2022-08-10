# 中文语音数据集
# 1. SpeechOceanDataset 最小的数据集，建议最先在这个上面测试
# 2. PrimeWordsDataset 优质的大型数据集 https://www.openslr.org/resources/47/
# 3. AiShellDataset 优质的大型数据集 http://www.openslr.org/33/
# 4. STCMDSDataset 优质的大型数据集 https://us.openslr.org/resources/38/
# 5. CvCorpus8Dataset 很差劲的firefox数据集，不要用这个，质量过差
# 6. AiShell3Dataset 优质的大型数据集 http://www.openslr.org/93/ 多说话人合成

import json
import os
from os.path import exists
import pandas as pd
import torch
import torchaudio
import pickle
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
# if __name__ == '__main__':
#     from audio import trim_mel_silence
# else:
#     from utils.audio import trim_mel_silence

cache_path = '/scratch/bh2283/.cache/'

class SpeechDataset(Dataset):
    def __init__(self, data_path, sample_rate=16000, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.threshold = 100000 # to avoid GPU memory used out
        self.batch_size = 128 # to avoid GPU memory used out
        self.split_ratio = [100, 5]
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass
    def split(self, split_ratio=None, seed=42):
        audio_dataset = self
        size = len(audio_dataset)
        my_split_ratio = self.split_ratio if split_ratio is None else split_ratio
        lengths = [(i*size)//sum(my_split_ratio) for i in my_split_ratio]
        lengths[-1] = size - sum(lengths[:-1])
        split_dataset = random_split(audio_dataset, lengths, generator=torch.Generator().manual_seed(seed))
        return split_dataset

class OpencpopDataset(SpeechDataset):

    def __init__(self, data_path, sample_rate=16000, transform=None):
        super().__init__(data_path, sample_rate, transform)
        transcript_file = data_path+'transcriptions.txt'
        self.transcript = self.gen_transcript(transcript_file)
        self.dataset_file_num = len(self.transcript)
        self.threshold = 120000 # to avoid GPU memory used out
        self.batch_size = 80 # to avoid GPU memory used out
        self.split_ratio = [1000, 3]

    def __len__(self):
        return self.dataset_file_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.dataset_file_num:
            return {'audio': None, 'text': None}
        line = self.transcript[idx]
        id, text, phoneme, note, note_duration, phoneme_duration, slur_note = self.parser_line(line)
        waveform = self.get_audio(id)
        # text_with_p, phoneme, note, note_duration = merge_note(text, phoneme, note, note_duration)
        sample = {'audio': waveform, 'text': line}
        if self.transform:
            sample = self.transform(sample, self.sample_rate)
        return sample

    def get_audio(self, id):
        wav_path = self.data_path+'wavs/'+str(id)+'.wav'
        waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform[0].unsqueeze(0), sample_rate, self.sample_rate)
        return waveform

    def parser_line(self, line):
        id, text, phoneme, note, note_duration, phoneme_duration, slur_note = line.split('|')
        phoneme = phoneme.split(' ')
        note = note.split(' ')
        note_duration = [float(i) for i in note_duration.split(' ')]
        phoneme_duration = [float(i) for i in phoneme_duration.split(' ')]
        slur_note = [int(i) for i in slur_note.split(' ')]
        assert len(phoneme) == len(note_duration) and len(phoneme_duration) == len(slur_note) and len(slur_note) == len(phoneme)
        return id, text, phoneme, note, note_duration, phoneme_duration, slur_note

    def gen_transcript(self, transcript_file):
        with open(transcript_file) as f:
            lines = f.read().split('\n')
            if (lines[-1]==''):
                lines = lines[:-1]
            return lines

    def split(self, split_ratio=None, seed=42):
        audio_dataset = self
        size = len(audio_dataset)
        my_split_ratio = self.split_ratio if split_ratio is None else split_ratio
        lengths = [(i*size)//sum(my_split_ratio) for i in my_split_ratio]
        lengths[-1] = size - sum(lengths[:-1])
        split_dataset = random_split(audio_dataset, lengths, generator=torch.Generator().manual_seed(seed))
        return split_dataset

class MusicLoaderGenerator:
    def __init__(self, 
        labels,
        num_workers=0,
        sample_rate = 44100,
        min_range = 512 * 6, # 默认删除过短的音频
        max_range = None, # 默认删除4秒以上长度的音频
        use_mel=True, # other than fft
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> None:
        self.sample_rate = sample_rate
        self.min_range = min_range
        if not max_range:
            max_range = self.sample_rate * 4 # 4 sec
        self.phoneme_labels, self.note_labels, self.slur_labels = labels
        self.phoneme_look_up = {s: i for i, s in enumerate(self.phoneme_labels)}
        self.note_look_up = {s: i for i, s in enumerate(self.note_labels)}
        self.slur_look_up = {s: i for i, s in enumerate(self.slur_labels)}
        self.device = device
        self.num_workers = num_workers
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,\
            n_fft=1024,power=1,hop_length=256,win_length=1024, n_mels=80, \
                f_min=0.0, f_max=8000.0, mel_scale="slaney", norm="slaney")
        self.fft = torchaudio.transforms.Spectrogram(n_fft=1024,hop_length=256,\
            win_length=None)
        self.use_mel = use_mel
        self.version = '0.01'

    def label2id(self, look_up, str):
        # if isinstance(str[0], list):
        #     return torch.stack([torch.tensor([look_up[i] for i in sub] if len(sub)==2 else [look_up[sub[0]], look_up['-']]) for sub in str])
        return torch.tensor([look_up[i] for i in str])

    def deep_label2id(self, look_up, str):
        return torch.stack([torch.tensor(\
            [look_up[i] for i in sub] if len(sub)==2 \
                else [look_up[sub[0]], look_up['-']]) for sub in str])
        
    def id2label(self, labels, idcs):
        return ''.join([labels[i] for i in idcs])

    def quant(self, duration):
        min_seg = 512/44100
        if duration < 2:
            return int(duration//min_seg)
        return int(2//min_seg + (duration-2)//(min_seg*2))

    def data_transfer(self, data):
        data_transferred = {
            'audio_duration': torch.tensor(data['audio_duration']).reshape([-1,1]),
            'chinese': data['chinese'], # 该音频汉字
            'phoneme': self.deep_label2id(self.phoneme_look_up, data['phoneme']), # 拼音
            'phoneme_pre': self.deep_label2id(self.phoneme_look_up, data['phoneme_pre']), # 前一个汉字的拼音
            'phoneme_post': self.deep_label2id(self.phoneme_look_up, data['phoneme_post']), # 后一个汉字的拼音
            'note': self.label2id(self.note_look_up, data['note']), # 音调音符
            'note_pre': self.label2id(self.note_look_up, data['note_pre']),
            'note_post': self.label2id(self.note_look_up, data['note_post']),
            'slur': self.label2id(self.slur_look_up, data['slur']), # 是否为延长音
        }
        return data_transferred

    def collate_wrapper(self, batch:list): # RAW
        bs = len(batch)
        sample_rate = self.sample_rate
        audio, audio_len, audio_duration, audio_duration_quant, chinese, phoneme,\
            phoneme_pre, phoneme_post, note, note_pre, note_post, slur,\
                mel, mel_len = [], [], [], [], [], [], [], [], [], [], [], [], [], []
        safe_log = lambda x: torch.log(x+2**(-15))
        for data in batch:
            audio_f = data['audio']
            chinese_f = data['chinese']
            phoneme_f = data['phoneme']
            note_f = data['note']
            duration_f = data['duration']
            duration_cum = duration_f.copy()
            for i in range(1, len(duration_cum)):
                duration_cum[i] += duration_cum[i-1]
            slur_f = data['slur']
            for i in range(len(chinese_f)):
                start = 0 if i == 0 else int(duration_cum[i-1]*sample_rate)
                end = int(duration_cum[i]*sample_rate)
                if end-start<self.min_range \
                    or audio_f.shape[-1]-start<self.min_range:
                    continue
                wave_chunk = audio_f[0, start: end]
                audio.append(wave_chunk)
                audio_len.append(len(wave_chunk))
                audio_duration.append(duration_f[i])
                audio_duration_quant.append(self.quant(duration_f[i]))
                chinese.append(chinese_f[i])
                phoneme.append(phoneme_f[i])
                phoneme_pre.append(phoneme_f[i-1]if i>0 else ['SP'])
                phoneme_post.append(phoneme_f[i+1]if i+1<len(phoneme_f) else ['SP'])
                note.append(note_f[i])
                note_pre.append(note_f[i-1]if i>0 else 'rest')
                note_post.append(note_f[i+1]if i+1<len(note_f) else 'rest')
                slur.append(slur_f[i])
                if self.use_mel:
                    mel_chunk = self.mel_transform(wave_chunk)
                    mel.append(safe_log(mel_chunk).transpose(0,1))
                else:
                    mel_chunk = self.fft(wave_chunk)
                    mel.append(safe_log(mel_chunk).transpose(0,1))
                mel_len.append(mel_chunk.shape[-1])
        # mel = pad_sequence(mel, batch_first=True, padding_value=torch.log(torch.tensor(2**(-15)))).permute(0,2,1)
        mel = pad_sequence(mel, batch_first=True, padding_value=0).permute(0,2,1)
        mel_len = torch.tensor(mel_len)
        
        return {
            'audio': audio,  # 单个字的raw音频
            'audio_len': audio_len, # 该音频数据长度
            'audio_duration': torch.tensor(audio_duration).reshape([-1,1]), # 真实音屏时间长度
            'audio_duration_quant': torch.tensor(audio_duration_quant), # 量化后音屏时间长度
            'chinese': chinese, # 该音频汉字
            'phoneme': self.deep_label2id(self.phoneme_look_up, phoneme), # 拼音
            'phoneme_pre': self.deep_label2id(self.phoneme_look_up, phoneme_pre), # 前一个汉字的拼音
            'phoneme_post': self.deep_label2id(self.phoneme_look_up, phoneme_post), # 后一个汉字的拼音
            'note': self.label2id(self.note_look_up, note), # 音调音符
            'note_pre': self.label2id(self.note_look_up, note_pre),
            'note_post': self.label2id(self.note_look_up, note_post),
            'slur': self.label2id(self.slur_look_up, slur), # 是否为延长音
            'mel': mel,
            'mel_len': mel_len
            }

    def dataloader(self, audioDataset, batch_size, shuffle=True):
        # k_size is the kernel size for the encoder, for data augmentation
        self.threshold = audioDataset.dataset.threshold
        return DataLoader(audioDataset, batch_size,
                            shuffle, num_workers=self.num_workers, collate_fn=self.collate_wrapper)



if __name__ == '__main__':
    from helper import parser_line, merge_note, get_pitch_labels, get_transposed_phoneme_labels, print_all
    def dataset_transform(sample, sample_rate=None):
        id, text, phoneme, note, note_duration, phoneme_duration, slur_note = parser_line(sample['text'])
        text_with_p, phoneme, note, note_duration, slur_note = merge_note(text, phoneme, note, note_duration, slur_note)
        sample['chinese'] = text_with_p
        sample['phoneme'] = phoneme
        sample['note'] = note
        sample['duration'] = note_duration
        sample['slur'] = slur_note
        return sample

    dataset = OpencpopDataset('/scratch/bh2283/data/opencpop/segments/', transform=dataset_transform, sample_rate=44100)
    train_set, test_set = dataset.split()
    note_labels = get_pitch_labels()
    phoneme_labels = get_transposed_phoneme_labels()
    slur_labels = [0, 1]

    labels = (
        phoneme_labels,
        note_labels,
        slur_labels
    )
    DATALOADER_WORKERS = 0
    loaderGenerator = MusicLoaderGenerator(labels, DATALOADER_WORKERS, use_mel=False)
    train_loader = loaderGenerator.dataloader(train_set, batch_size=8)
    print('train_set:', len(train_set), 'test_set:',len(test_set))
    steps = 1
    for i_batch, sample_batched in enumerate(train_loader):
        if steps <= 0:
            break
        print(sample_batched['mel'].shape)
        print(sample_batched['chinese'], sample_batched['mel_len'][:3], sample_batched['audio_len'][:3], sample_batched['audio_duration'][:3], )
        steps -= 1

# 中文语音数据集
# 1. SpeechOceanDataset 最小的数据集，建议最先在这个上面测试
# 2. PrimeWordsDataset 优质的大型数据集 https://www.openslr.org/resources/47/
# 3. AiShellDataset 优质的大型数据集 http://www.openslr.org/33/
# 4. STCMDSDataset 优质的大型数据集 https://us.openslr.org/resources/38/
# 5. CvCorpus8Dataset 很差劲的firefox数据集，不要用这个，质量过差

import json
import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split


class SpeechOceanDataset(Dataset):

    def __init__(self, data_path, sample_rate=16000, transform=None):
        meta_data = data_path + 'metadata.csv'
        self.meta_data = pd.read_csv(meta_data, sep='\t')
        self.dataset_file_num = len(self.meta_data)
        self.data_path = data_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.threshold = 100000 # to avoid GPU memory used out
        self.batch_size = 128 # to avoid GPU memory used out
        self.split_ratio = [100, 5]

    def __len__(self):
        return self.dataset_file_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.dataset_file_num:
            return {'audio': None, 'text': None}
        audio_name = os.path.join(self.data_path,
                                  self.meta_data.iloc[idx, 0])
        waveform, sample_rate = torchaudio.load(audio_name)
        audio_content = self.meta_data.iloc[idx, 1]
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        sample = {'audio': waveform, 'text': audio_content}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def split(self, split_ratio=None, seed=42):
        audio_dataset = self
        size = len(audio_dataset)
        my_split_ratio = self.split_ratio if split_ratio is None else split_ratio
        lengths = [(i*size)//sum(my_split_ratio) for i in my_split_ratio]
        lengths[-1] = size - sum(lengths[:-1])
        split_dataset = random_split(audio_dataset, lengths, generator=torch.Generator().manual_seed(seed))
        return split_dataset

class PrimeWordsDataset(Dataset):

    def __init__(self, data_path, sample_rate=16000, transform=None):
        with open(data_path+'set1_transcript.json') as f:
            json_data = json.load(f)
        self.json_data = json_data
        # self.wav_files = self.get_all_wav_files(data_path, self.transcript)
        self.dataset_file_num = len(self.json_data)
        self.data_path = data_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.threshold = 220000 # to avoid GPU memory used out
        self.batch_size = 40 # to avoid GPU memory used out
        self.split_ratio = [1000, 8]

    def __len__(self):
        return self.dataset_file_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.dataset_file_num:
            return {'audio': None, 'text': None}
        audio_file, audio_content = self.parse_line(self.json_data[idx])
        waveform, sample_rate = torchaudio.load(self.get_wav(audio_file))
        waveform = waveform
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        sample = {'audio': waveform, 'text': audio_content}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def parse_line(self, line):
        return line['file'], line['text']

    def get_wav(self, file_name):
        path = self.data_path+'audio_files/'+file_name[0]+'/'+file_name[:2]+'/'+file_name
        return path

    def split(self, split_ratio=None, seed=42):
        audio_dataset = self
        size = len(audio_dataset)
        my_split_ratio = self.split_ratio if split_ratio is None else split_ratio
        lengths = [(i*size)//sum(my_split_ratio) for i in my_split_ratio]
        lengths[-1] = size - sum(lengths[:-1])
        split_dataset = random_split(audio_dataset, lengths, generator=torch.Generator().manual_seed(seed))
        return split_dataset

class AiShellDataset(Dataset):

    def __init__(self, data_path, sample_rate=16000, transform=None):
        transcript_file = data_path+'transcript/aishell_transcript_v0.8.txt'
        self.transcript = self.gen_transcript(transcript_file)
        self.wav_files = self.get_all_wav_files(data_path, self.transcript)
        self.dataset_file_num = len(self.wav_files)
        self.data_path = data_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.threshold = 120000 # to avoid GPU memory used out
        self.batch_size = 80 # to avoid GPU memory used out
        self.split_ratio = [1000, 5]

    def __len__(self):
        return self.dataset_file_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.dataset_file_num:
            return {'audio': None, 'text': None}
        audio_name = self.wav_files[idx]
        waveform, sample_rate = torchaudio.load(audio_name)
        waveform = waveform
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        dict_id = audio_name.rsplit('/',1)[-1].split('.')[0]
        audio_content = self.transcript[dict_id]
        sample = {'audio': waveform, 'text': audio_content}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def parse_line(self, line):
        id, text = line.split(' ', 1)
        text = ''.join(text.split(' '))
        return id, text

    def gen_transcript(self, transcript_file):
        transcript = {}
        with open(transcript_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')[:-1]
            for line in lines:
                id, text = self.parse_line(line)
                transcript[id] = text
        return transcript

    def get_all_wav_files(self, path, transcript):
        folders = []
        train = os.listdir(path+'wav/train/')
        folders += [path+'wav/train/'+i for i in train]
        dev = os.listdir(path+'wav/dev/')
        folders += [path+'wav/dev/'+i for i in dev]
        test = os.listdir(path+'wav/test/')
        folders += [path+'wav/test/'+i for i in test]
        files = []
        for folder in folders:
            files += [folder+'/'+i for i in os.listdir(folder) if i[:-4] in transcript]
        return files
    
    def split(self, split_ratio=None, seed=42):
        audio_dataset = self
        size = len(audio_dataset)
        my_split_ratio = self.split_ratio if split_ratio is None else split_ratio
        lengths = [(i*size)//sum(my_split_ratio) for i in my_split_ratio]
        lengths[-1] = size - sum(lengths[:-1])
        split_dataset = random_split(audio_dataset, lengths, generator=torch.Generator().manual_seed(seed))
        return split_dataset

class STCMDSDataset(Dataset):

    def __init__(self, data_path, sample_rate=16000, transform=None):
        files = os.listdir(data_path)
        file_names = []
        for file in files:
            if file.split('.')[1] == 'txt':
                file_names.append(file.split('.')[0])
        self.dataset_file_num = len(file_names)
        self.file_names = file_names
        self.data_path = data_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.threshold = 90000 # to avoid GPU memory used out
        self.batch_size = 128 # to avoid GPU memory used out
        self.split_ratio = [1000, 5]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = self.get_audio(idx)
        waveform, sample_rate = torchaudio.load(audio_name)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        audio_content = self.get_text(idx)
        sample = {'audio': waveform, 'text': audio_content}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_audio(self, x): 
        return self.data_path+self.file_names[x]+'.wav' if x < self.dataset_file_num else None
        
    def get_text(self, x): 
        return open(self.data_path+self.file_names[x]+'.txt', "r").read() if x < self.dataset_file_num else None
    
    def split(self, split_ratio=None, seed=42):
        audio_dataset = self
        size = len(audio_dataset)
        my_split_ratio = self.split_ratio if split_ratio is None else split_ratio
        lengths = [(i*size)//sum(my_split_ratio) for i in my_split_ratio]
        lengths[-1] = size - sum(lengths[:-1])
        split_dataset = random_split(audio_dataset, lengths, generator=torch.Generator().manual_seed(seed))
        return split_dataset
        
AudioDataset = STCMDSDataset

class CvCorpus8Dataset(Dataset):

    def __init__(self, data_path, sample_rate=16000, transform=None):
        df1 = pd.read_csv(data_path+'validated.tsv',sep='\t')[['path', 'sentence']]
        # df2 = pd.read_csv(data_path+'invalidated.tsv',sep='\t')[['path', 'sentence']]
        # df3 = pd.read_csv(data_path+'other.tsv',sep='\t')[['path', 'sentence']]
        # df = pd.concat([df1, df2, df3])
        df = df1
        audio_path = df['path'].to_list()
        sentence_text = df['sentence'].to_list()
        assert len(audio_path) == len(sentence_text)
        self.audio_path = audio_path
        self.sentence_text = sentence_text
        self.size = len(audio_path)
        self.data_path = data_path
        self.transform = transform
        self.sample_rate = sample_rate
        self.threshold = 170000 # to avoid GPU memory used out
        self.batch_size = 64 # to avoid GPU memory used out
        self.split_ratio = [100, 1]

    def __len__(self):
        return len(self.audio_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = self.get_audio(idx)
        waveform, sample_rate = torchaudio.load(audio_name)
        waveform = waveform
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        audio_content = self.get_text(idx)
        sample = {'audio': waveform, 'text': audio_content}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_audio(self, x): 
        return self.data_path+'clips/'+self.audio_path[x] if x < len(self) else None
        
    def get_text(self, x): 
        return self.sentence_text[x] if x < len(self) else None
    
    def split(self, split_ratio=None, seed=42):
        audio_dataset = self
        size = len(audio_dataset)
        my_split_ratio = self.split_ratio if split_ratio is None else split_ratio
        lengths = [(i*size)//sum(my_split_ratio) for i in my_split_ratio]
        lengths[-1] = size - sum(lengths[:-1])
        split_dataset = random_split(audio_dataset, lengths, generator=torch.Generator().manual_seed(seed))
        return split_dataset

class LoaderGenerator:
    def __init__(self, labels, chinese2pinyin, k_size=0) -> None:
        self.k_size = k_size
        self.labels = labels
        self.look_up = {s: i for i, s in enumerate(labels)}
        self.chinese2pinyin = chinese2pinyin
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k_size = 13 # set a small integer, should be encoder kernel size, will be modidfied in dataloader

    def label2id(self, str):
        return [self.look_up[i] for i in str]

    def id2label(self, idcs):
        return ''.join([self.labels[i] for i in idcs])

    def batch_filter(self, batch:list):
        # remove all audio with tag if audio length > threshold
        for i in range(len(batch)-1, -1, -1):
            if batch[i]['audio'].shape[-1] > self.threshold:
                del batch[i]
        return batch

    def collate_wrapper(self, batch:list):
        batch = self.batch_filter(batch)
        bs = len(batch)
        rand_shift = torch.randint(self.k_size, (bs,))
        audio_list = [batch[i]['audio'][:,rand_shift[i]:] for i in range(bs)]
        audio_length = torch.tensor([audio.shape[-1] for audio in audio_list])
        max_audio_length = torch.max(audio_length)
        audio_list = torch.cat([
            torch.cat(
            (audio, torch.zeros(max_audio_length-audio.shape[-1]).unsqueeze(0)), -1)
            for audio in audio_list], 0)
        target_list = [self.label2id(self.chinese2pinyin(item['text'])) for item in batch]
        target_length = torch.tensor([len(l) for l in target_list])
        max_target_length = torch.max(target_length)
        target_list = torch.cat([
            torch.cat(
            (torch.tensor(l), torch.zeros([max_target_length-len(l)], dtype=torch.int)), -1).unsqueeze(0) 
            for l in target_list], 0)
        device = self.device
        return {'audio': audio_list.to(device), 'target': target_list.to(device), 'audio_len': audio_length.to(device), 'target_len': target_length.to(device)}

    def dataloader(self, audioDataset, batch_size, shuffle=True):
        # k_size is the kernel size for the encoder, for data augmentation
        self.threshold = audioDataset.dataset.threshold
        return DataLoader(audioDataset, batch_size,
                            shuffle, num_workers=0, collate_fn=self.collate_wrapper)

if __name__ == '__main__':
    # dataset = AudioDataset('./data/ST-CMDS-20170001_1-OS/')
    # dataset = CvCorpus8Dataset('./data/cv-corpus-8.0-2022-01-19/zh-CN/')
    # dataset = AiShellDataset('./data/data_aishell/')
    # dataset = PrimeWordsDataset('./data/primewords_md_2018_set1/')
    dataset = SpeechOceanDataset('./data/zhspeechocean/')
    batch_size = 8
    train_set, test_set = dataset.split()
    k_size = 5 # kernel size for audio encoder
    from pypinyin import lazy_pinyin
    def chinese2pinyin(text):
        pinyin = lazy_pinyin(text, strict=True,errors=lambda x: u'')
        pinyin = [i for i in '|'.join(pinyin)]
        return pinyin
    from helper import get_labels
    labels = get_labels()
    loaderGenerator = LoaderGenerator(labels, chinese2pinyin, k_size)
    train_loader = loaderGenerator.dataloader(train_set, batch_size)
    test_loader = loaderGenerator.dataloader(test_set, batch_size) # without backprop, can use larger batch
    print('train_set:', len(train_set), 'test_set:',len(test_set))
    steps = 10
    for i_batch, sample_batched in enumerate(test_loader):
        print(sample_batched['audio'].shape, sample_batched['target'].shape)
        # for i in sample_batched['audio']:
        #     print(i.shape)
        if steps < 0:
            break
        steps -= 1

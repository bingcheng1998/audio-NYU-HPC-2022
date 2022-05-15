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
if __name__ == '__main__':
    from audio import trim_mel_silence
else:
    from utils.audio import trim_mel_silence

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

class SpeechOceanDataset(SpeechDataset):
    def __init__(self, data_path, sample_rate=16000, transform=None):
        super().__init__(data_path, sample_rate, transform)
        meta_data = data_path + 'metadata.csv'
        self.meta_data = pd.read_csv(meta_data, sep='\t')
        self.dataset_file_num = len(self.meta_data)
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
            sample = self.transform(sample, self.sample_rate)
        return sample

class PrimeWordsDataset(SpeechDataset):

    def __init__(self, data_path, sample_rate=16000, transform=None):
        super().__init__(data_path, sample_rate, transform)
        with open(data_path+'set1_transcript.json') as f:
            json_data = json.load(f)
        self.json_data = json_data
        self.dataset_file_num = len(self.json_data)
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
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        sample = {'audio': waveform, 'text': audio_content}
        if self.transform:
            sample = self.transform(sample, self.sample_rate)
        return sample

    def parse_line(self, line):
        return line['file'], line['text']

    def get_wav(self, file_name):
        path = self.data_path+'audio_files/'+file_name[0]+'/'+file_name[:2]+'/'+file_name
        return path

class AiShellDataset(SpeechDataset):

    def __init__(self, data_path, sample_rate=16000, transform=None):
        super().__init__(data_path, sample_rate, transform)
        transcript_file = data_path+'transcript/aishell_transcript_v0.8.txt'
        self.transcript = self.gen_transcript(transcript_file)
        self.wav_files = self.get_all_wav_files(data_path, self.transcript)
        self.dataset_file_num = len(self.wav_files)
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
            sample = self.transform(sample, self.sample_rate)
        return sample

    def parse_line(self, line):
        id, text = line.split(' ', 1)
        text = ''.join(text.split(' '))
        return id, text

    def gen_transcript(self, transcript_file):
        pk = cache_path+'dataset_temp/aishell_transcript.pickle'
        if exists(pk):
            with open(pk,"rb") as f:
                return pickle.load(f)
        transcript = {}
        with open(transcript_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')[:-1]
            for line in lines:
                id, text = self.parse_line(line)
                transcript[id] = text
        with open(pk,"wb") as f:
            pickle.dump(transcript, f)
        return transcript

    def get_all_wav_files(self, path, transcript):
        pk = cache_path+'dataset_temp/aishell_all_wav_files.pickle'
        if exists(pk):
            with open(pk,"rb") as f:
                return pickle.load(f)
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
        with open(pk,"wb") as f:
            pickle.dump(files, f)
        return files
    
    def split(self, split_ratio=None, seed=42):
        audio_dataset = self
        size = len(audio_dataset)
        my_split_ratio = self.split_ratio if split_ratio is None else split_ratio
        lengths = [(i*size)//sum(my_split_ratio) for i in my_split_ratio]
        lengths[-1] = size - sum(lengths[:-1])
        split_dataset = random_split(audio_dataset, lengths, generator=torch.Generator().manual_seed(seed))
        return split_dataset

class AiShell3Dataset(SpeechDataset):

    def __init__(self, data_path, sample_rate=16000, transform=None):
        super().__init__(data_path, sample_rate, transform)
        transcript_file = data_path+'content.txt'
        self.transcript = self.gen_transcript(transcript_file)
        self.wav_files = self.get_all_wav_files(data_path, self.transcript)
        self.dataset_file_num = len(self.wav_files)
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
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        dict_id = audio_name.rsplit('/',1)[-1].split('.')[0]
        audio_content = self.transcript[dict_id]
        sample = {'audio': waveform, 'text': audio_content, 'audio_path': audio_name}
        if self.transform:
            sample = self.transform(sample, self.sample_rate)
        return sample

    def parse_line(self, line):
        id, text = line.split('\t', 1)
        id = id.split('.')[0]
        # text = ''.join(text.split(' '))
        return id, text

    def gen_transcript(self, transcript_file):
        pk = cache_path+'dataset_temp/aishell3transcript.pickle'
        if exists(pk):
            with open(pk,"rb") as f:
                return pickle.load(f)
        transcript = {}
        with open(transcript_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')[:-1]
            for line in lines:
                id, text = self.parse_line(line)
                transcript[id] = text
        with open(pk,"wb") as f:
            pickle.dump(transcript, f)
        return transcript

    def get_all_wav_files(self, path, transcript):
        pk = cache_path+'dataset_temp/aishell3_all_wav_files.pickle'
        if exists(pk):
            with open(pk,"rb") as f:
                return pickle.load(f)
        people_folder = path+'wav/'
        wav_folders = [people_folder+i for i in os.listdir(people_folder)]
        files = []
        for wav_folder in wav_folders:
            files += [wav_folder+'/'+i for i in os.listdir(wav_folder) if i[:-4] in transcript]
        with open(pk,"wb") as f:
            pickle.dump(files, f)
        return files
    
    def split(self, split_ratio=None, seed=42):
        audio_dataset = self
        size = len(audio_dataset)
        my_split_ratio = self.split_ratio if split_ratio is None else split_ratio
        lengths = [(i*size)//sum(my_split_ratio) for i in my_split_ratio]
        lengths[-1] = size - sum(lengths[:-1])
        split_dataset = random_split(audio_dataset, lengths, generator=torch.Generator().manual_seed(seed))
        return split_dataset

class AiShell3PersonDataset(SpeechDataset):

    def __init__(self, data_path, person_id, sample_rate=16000, transform=None):
        super().__init__(data_path, sample_rate, transform)
        transcript_file = data_path+'content.txt'
        self.transcript = self.gen_transcript(transcript_file)
        self.wav_files = self.get_all_wav_files(data_path, self.transcript, person_id)
        self.dataset_file_num = len(self.wav_files)
        self.person_id = person_id
        self.threshold = 120000 # to avoid GPU memory used out
        self.batch_size = 80 # to avoid GPU memory used out
        self.split_ratio = [1000, 0]

    def __len__(self):
        return self.dataset_file_num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.dataset_file_num:
            return {'audio': None, 'text': None}
        audio_name = self.wav_files[idx]
        waveform, sample_rate = torchaudio.load(audio_name)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        dict_id = audio_name.rsplit('/',1)[-1].split('.')[0]
        audio_content = self.transcript[dict_id]
        sample = {'audio': waveform, 'text': audio_content}
        if self.transform:
            sample = self.transform(sample, self.sample_rate)
        return sample

    def parse_line(self, line):
        id, text = line.split('\t', 1)
        id = id.split('.')[0]
        # text = ''.join(text.split(' '))
        return id, text

    def gen_transcript(self, transcript_file):
        pk = cache_path+'dataset_temp/aishell3transcript.pickle'
        if exists(pk):
            with open(pk,"rb") as f:
                return pickle.load(f)
        transcript = {}
        with open(transcript_file, 'r') as f:
            content = f.read()
            lines = content.split('\n')[:-1]
            for line in lines:
                id, text = self.parse_line(line)
                transcript[id] = text
        with open(pk,"wb") as f:
            pickle.dump(transcript, f)
        return transcript

    def get_all_wav_files(self, path, transcript, person_id):
        wav_folder = path+f'wav/{person_id}/'
        files = [wav_folder+'/'+i for i in os.listdir(wav_folder) if i[:-4] in transcript]
        return files
    
    def split(self, split_ratio=None, seed=42):
        audio_dataset = self
        size = len(audio_dataset)
        my_split_ratio = self.split_ratio if split_ratio is None else split_ratio
        lengths = [(i*size)//sum(my_split_ratio) for i in my_split_ratio]
        lengths[-1] = size - sum(lengths[:-1])
        split_dataset = random_split(audio_dataset, lengths, generator=torch.Generator().manual_seed(seed))
        return split_dataset

class STCMDSDataset(SpeechDataset):

    def __init__(self, data_path, sample_rate=16000, transform=None):
        super().__init__(data_path, sample_rate, transform)
        files = os.listdir(data_path)
        file_names = []
        for file in files:
            if file.split('.')[1] == 'txt':
                file_names.append(file.split('.')[0])
        self.dataset_file_num = len(file_names)
        self.file_names = file_names
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
            sample = self.transform(sample, self.sample_rate)
        return sample

    def get_audio(self, x): 
        return self.data_path+self.file_names[x]+'.wav' if x < self.dataset_file_num else None
        
    def get_text(self, x): 
        return open(self.data_path+self.file_names[x]+'.txt', "r").read() if x < self.dataset_file_num else None

class CvCorpus8Dataset(SpeechDataset):

    def __init__(self, data_path, sample_rate=16000, transform=None):
        super().__init__(data_path, sample_rate, transform)
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
            sample = self.transform(sample, self.sample_rate)
        return sample

    def get_audio(self, x): 
        return self.data_path+'clips/'+self.audio_path[x] if x < len(self) else None
        
    def get_text(self, x): 
        return self.sentence_text[x] if x < len(self) else None

class MelLoaderGenerator:
    def __init__(self, 
        labels, k_size=0, 
        num_workers = 0,
        sample_rate = 16000,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> None:
        self.k_size = k_size
        self.labels = labels
        self.look_up = {s: i for i, s in enumerate(labels)}
        self.device = device
        self.num_workers = num_workers
        self.sample_rate = sample_rate

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

    def collate_wrapper(self, batch:list): # RAW
        batch = self.batch_filter(batch)
        bs = len(batch)
        rand_shift = torch.randint(self.k_size, (bs,))
        audio_list = [batch[i]['audio'][:,rand_shift[i]:] for i in range(bs)]
        audio_length = [audio.shape[-1] for audio in audio_list]
        target_list = [self.label2id(item['text']) for item in batch]
        target_length = [len(l) for l in target_list]
        def path2user(path):
            # eg: /scratch/bh2283/data/data_aishell3/train/wav/SSB0145/SSB01450373.wav to SSB0145
            return path.rsplit('/', 1)[0].rsplit('/', 1)[1]
        speaker = [path2user(batch[i]['audio_path']) for i in range(bs)]

        target_length, target_list, audio_length, audio_list = zip(*sorted(zip(target_length, target_list, audio_length, audio_list), reverse=True))
        target_length = torch.tensor(target_length)
        audio_length = torch.tensor(audio_length)

        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,\
            n_fft=1024,power=1,hop_length=256,win_length=1024, n_mels=80, \
                f_min=0.0, f_max=8000.0, mel_scale="slaney", norm="slaney")
        safe_log = lambda x: torch.log(x+2**(-15))

        # for audio in audio_list:
            # print(audio.shape)

        mels_list = [safe_log(trim_mel_silence(mel_transform(audio_list[i]).squeeze())).transpose(0,1) for i in range(len(audio_list))]
        # for mel in mels_list:
        #     print(mel.shape)
        mel_length = torch.tensor([mel.shape[-2] for mel in mels_list])
        mels_tensor = pad_sequence(mels_list, batch_first=True, padding_value=torch.log(torch.tensor(2**(-15)))).permute(0,2,1)
        # print(mels_tensor.shape) # [bs, mel_bins, L]

        # max_audio_length = torch.max(audio_length)
        # audio_list = torch.cat([
        #     torch.cat(
        #     (audio, torch.zeros(max_audio_length-audio.shape[-1]).unsqueeze(0)), -1)
        #     for audio in audio_list], 0)
        
        max_target_length = torch.max(target_length)
        target_list = torch.cat([
            torch.cat(
            (torch.tensor(l), torch.zeros([max_target_length-len(l)], dtype=torch.int)), -1).unsqueeze(0) 
            for l in target_list], 0)
        # device = self.device
        return {
                # 'audio': audio_list, 'audio_len': audio_length, 
                'target': target_list, 'target_len': target_length,
                'mel': mels_tensor, 'mel_len': mel_length, 'speaker': speaker
                }

    def dataloader(self, audioDataset, batch_size, shuffle=True):
        # k_size is the kernel size for the encoder, for data augmentation
        self.threshold = audioDataset.dataset.threshold
        return DataLoader(audioDataset, batch_size,
                            shuffle, num_workers=self.num_workers, collate_fn=self.collate_wrapper)

class RawLoaderGenerator:
    def __init__(self, 
        labels, k_size=0, 
        num_workers=0,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> None:
        self.k_size = k_size
        self.labels = labels
        self.look_up = {s: i for i, s in enumerate(labels)}
        self.device = device
        self.num_workers = num_workers
        self.version = '0.02'

    def label2id(self, str):
        return [self.look_up[i] for i in str]

    def id2label(self, idcs):
        return ''.join([self.labels[i] for i in idcs])

    def batch_filter(self, batch:list):
        # remove all audio with tag if audio length > threshold
        for i in range(len(batch)-1, -1, -1):
            if batch[i]['audio'].shape[-1] > self.threshold: # 256 is the hop_length of fft
                del batch[i]
        return batch

    def collate_wrapper(self, batch:list): # RAW
        batch = self.batch_filter(batch)
        bs = len(batch)
        rand_shift = torch.randint(self.k_size, (bs,))
        audio_list = [batch[i]['audio'][:,rand_shift[i]:] for i in range(bs)]
        audio_length = [audio.shape[-1] for audio in audio_list]
        target_list = [self.label2id(item['text']) for item in batch]
        target_length = [len(l) for l in target_list]

        target_length, target_list, audio_length, audio_list = zip(*sorted(zip(target_length, target_list, audio_length, audio_list), reverse=True))
        target_length = torch.tensor(target_length)
        audio_length = torch.tensor(audio_length)

        max_audio_length = torch.max(audio_length)
        audio_list = torch.cat([
            torch.cat(
            (audio, torch.zeros(max_audio_length-audio.shape[-1]).unsqueeze(0)), -1)
            for audio in audio_list], 0)
        
        max_target_length = torch.max(target_length)
        target_list = torch.cat([
            torch.cat(
            (torch.tensor(l), torch.zeros([max_target_length-len(l)], dtype=torch.int)), -1).unsqueeze(0) 
            for l in target_list], 0)
        device = self.device
        return {'audio': audio_list, 'audio_len': audio_length, 
                'target': target_list, 'target_len': target_length}

    def dataloader(self, audioDataset, batch_size, shuffle=True):
        # k_size is the kernel size for the encoder, for data augmentation
        self.threshold = audioDataset.dataset.threshold
        return DataLoader(audioDataset, batch_size,
                            shuffle, num_workers=self.num_workers, collate_fn=self.collate_wrapper)

if __name__ == '__main__':
    # def mel_audio_transform(sample, sample_rate):
    #     audio = sample['audio']
    #     text = sample['text']
    #     mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,\
    #         n_fft=1024,power=1,hop_length=256,win_length=1024, n_mels=80, \
    #             f_min=0.0, f_max=8000.0, mel_scale="slaney", norm="slaney")
    #     def chinese2pinyin(text):
    #         pinyin = lazy_pinyin(text, strict=True,errors=lambda x: u'')
    #         pinyin = [i for i in '|'.join(pinyin)]
    #         return pinyin
    #     safe_log = lambda x: torch.log(x+2**(-15))
    #     return {'audio':safe_log(mel_transform(audio)),
    #             'text': chinese2pinyin(text),
    #             'chinese': text}
    def ai_shell_3_transform(sample, sample_rate=None):
        audio = sample['audio']
        # audio = torchaudio.functional.vad(audio, sample_rate, trigger_level=5)
        audio = audio / torch.abs(audio).max()*0.15
        text = sample['text']
        text = text.split(' ')
        pinyin = [text[i] for i in range(len(text)) if i%2==1]
        pinyin = ' '.join(pinyin) # 使用空格分离单字
        chinese = [text[i] for i in range(len(text)) if i%2==0]
        sample['audio'] = audio
        sample['text'] = pinyin+' .'
        sample['chinese'] = chinese
        return sample

    def raw_audio_transform(sample, sample_rate=None):
        audio = sample['audio']
        audio = audio / torch.abs(audio).max()*0.15
        sample['audio'] = audio
        sample['chinese'] = sample['text']
        return sample
    # dataset = SpeechOceanDataset('/scratch/bh2283/data/zhspeechocean/', transform=raw_audio_transform)
    dataset = STCMDSDataset('/ST-CMDS-20170001_1-OS/', transform=raw_audio_transform)
    # dataset = CvCorpus8Dataset('/scratch/bh2283/data/cv-corpus-8.0-2022-01-19/zh-CN/', transform=raw_audio_transform)
    # dataset = AiShellDataset('/scratch/bh2283/data/data_aishell/', transform=raw_audio_transform)
    # dataset = PrimeWordsDataset('/scratch/bh2283/data/primewords_md_2018_set1/', transform=raw_audio_transform)
    # dataset = AiShell3Dataset('/scratch/bh2283/data/data_aishell3/train/', transform=ai_shell_3_transform)
    # dataset = AiShell3PersonDataset('/scratch/bh2283/data/data_aishell3/train/', transform=raw_audio_transform, person_id='SSB0011')
    from pypinyin import lazy_pinyin
    from helper import get_alphabet_labels as get_labels
    labels = get_labels()+('1','2','3','4','5',' ','.')
    loaderGenerator = MelLoaderGenerator(labels, k_size=256)
    # loaderGenerator = RawLoaderGenerator(labels, k_size=5)
    train_set, test_set = dataset.split()
    train_loader = loaderGenerator.dataloader(train_set, batch_size=8)
    print('train_set:', len(train_set), 'test_set:',len(test_set))
    steps = 10
    # for i_batch, sample_batched in enumerate(train_loader):
    #     if steps <= 0:
    #         break
    #     print(sample_batched['mel'].shape, sample_batched['target'].shape)
    #     print(sample_batched['mel_len'], sample_batched['target_len'])
    #     print(sample_batched['speaker'])
    #     steps -= 1

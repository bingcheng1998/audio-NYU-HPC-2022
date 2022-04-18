import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio

class AudioDataset(Dataset):

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

    def __len__(self):
        return len(self.file_names)

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
        return self.data_path+self.file_names[x]+'.wav' if x < self.dataset_file_num else None
        
    def get_text(self, x): 
        return open(self.data_path+self.file_names[x]+'.txt', "r").read() if x < self.dataset_file_num else None
    
    def split(self, split_ratio=[8, 2], seed=42):
        audio_dataset = self
        size = len(audio_dataset)
        lengths = [(i*size)//sum(split_ratio) for i in split_ratio]
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

    def collate_wrapper(self, batch):
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
            (torch.tensor(l), torch.zeros(max_target_length-len(l))), -1).unsqueeze(0) 
            for l in target_list], 0)
        device = self.device
        return {'audio': audio_list.to(device), 'target': target_list.to(device), 'audio_len': audio_length.to(device), 'target_len': target_length.to(device)}

    def dataloader(self, audioDataset, batch_size):
        # k_size is the kernel size for the encoder, for data augmentation
        return DataLoader(audioDataset, batch_size,
                            shuffle=True, num_workers=0, collate_fn=self.collate_wrapper)
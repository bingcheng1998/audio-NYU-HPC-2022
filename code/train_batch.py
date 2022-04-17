import os
data_path = './data/ST-CMDS-20170001_1-OS/'
files = os.listdir(data_path)
file_names = []
for file in files:
    if file.split('.')[1] == 'txt':
        file_names.append(file.split('.')[0])
dataset_file_num = len(file_names)
dataset_file_num

get_audio = lambda x: data_path+file_names[x]+'.wav' if x < dataset_file_num else None
get_text = lambda x: open(data_path+file_names[x]+'.txt', "r").read() if x < dataset_file_num else None

import random
# from IPython.display import Audio

rand_id = random.randint(0, 2400-1)

audio_file = get_audio(rand_id)

sentence = get_text(rand_id)

print(rand_id, sentence)

# import os
# from dataclasses import dataclass
# import IPython
# import matplotlib
# import matplotlib.pyplot as plt
import requests
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchaudio

# matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.__version__)
print(torchaudio.__version__)
print(device)

SPEECH_URL = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
SPEECH_FILE = "_assets/speech.wav"

if not os.path.exists(SPEECH_FILE):
    os.makedirs("_assets", exist_ok=True)
    with open(SPEECH_FILE, "wb") as file:
        file.write(requests.get(SPEECH_URL).content)

bundle = torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_EN
model = bundle.get_model().to(device)
labels = bundle.get_labels()
model.aux
# look_up = {s: i for i, s in enumerate(labels)} # 字母转数字

with torch.inference_mode():
    waveform, _ = torchaudio.load(SPEECH_FILE)
    emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()

print(model.aux, emission.shape)

"""## Do CTC Decoder"""


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


class NaiveCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        # indices = torch.unique_consecutive(indices, dim=-1)
        # indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


labels = bundle.get_labels()
# labels = list(labels)
look_up = {s: i for i, s in enumerate(labels)}  # 字母转数字
# # labels = {i: s for i, s in enumerate(labels)} # 数字转字母

# labels

# indices = torch.argmax(emission, dim=-1)  # [num_seq,]
# torch.unique_consecutive(indices, dim=0)

decoder = GreedyCTCDecoder(labels=labels)
transcript = decoder(emission)
print(transcript)

# ---------------------- Chinese -----------------------

# initial_table = ['b', 'p', 'm', 'f',
#                 'd', 't', 'n', 'l',
#                 'g', 'k', 'h',
#                 'j', 'q', 'x',
#                 'zh', 'ch', 'sh', 'r', 'z', 'c', 's']

# finals_table = [['i', 'u', 'v'], # 可以与下面的配成 iao, ue
#                 ['a', 'o', 'e', 
#                 'ai', 'ei', 'ao',
#                 'ou', 'an', 'en',
#                 'ang', 'eng', 'ong', 'er']]

# labels = ['-','|']+initial_table + finals_table[0] + finals_table[1]
# look_up = {s: i for i, s in enumerate(labels)} # 字母转数字

# decoder = GreedyCTCDecoder(labels=labels)

"""# Transfer Learning in Chinese"""

# ! pip install pypinyin

# import pypinyin
from pypinyin import lazy_pinyin, Style


# def chinese2pinyin(text):
#     initials = lazy_pinyin(text, strict=True, style=Style.INITIALS, errors=lambda x: u'')
#     finals = lazy_pinyin(text, strict=True, style=Style.FINALS, errors=lambda x: u'')
#     pinyin = ''
#     for i in range(len(finals)):
#         pinyin+='|'
#         if (initials[i] == '-'):
#             continue
#         pinyin+=initials[i]
#         pinyin+=finals[i]
#         if finals[i] == '':
#             pinyin+='n'
#     if pinyin[-1] == '|':
#         pinyin = pinyin[:-1]
#     return pinyin.lower().replace('w','u')

def chinese2pinyin(text):
    pinyin = lazy_pinyin(text, strict=True,errors=lambda x: u'')
    pinyin = [i for i in '|'.join(pinyin)]
    return ['|']+pinyin

print(''.join(chinese2pinyin("绿色的温水，迂回的乌烟，流过。啊！哇！妞儿归去！")))


def label2id(str):
    return [look_up[i] for i in str]


def id2label(idcs):
    return ''.join([labels[i] for i in idcs])


print(
    label2id(chinese2pinyin("美好，我很开心！")), ','.join(id2label(label2id(chinese2pinyin("美好，我很开心！"))))
)

from torch.utils.data import Dataset, DataLoader

sr = 16000
class AudioDataset(Dataset):

    def __init__(self, file_names, data_path, sample_rate=16000, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_names = file_names
        self.root_dir = data_path
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = get_audio(idx)
        waveform, sample_rate = torchaudio.load(audio_name)
        waveform = waveform
        
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        
        audio_content = get_text(idx)
        
        sample = {'audio': waveform, 'text': audio_content}

        if self.transform:
            sample = self.transform(sample)

        return sample

audio_dataset = AudioDataset(file_names, data_path, sample_rate=16000)
print('audio_dataset[0]', audio_dataset[0])

#---------------- Adapt Model -----------------------

print(model.aux)

model = bundle.get_model()
for param in model.parameters():
    param.requires_grad = False
model.aux = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features=model.aux.in_features, out_features=len(labels), bias=True)
)
# model.aux = nn.Linear(in_features=model.aux.in_features, out_features=len(labels), bias=True)
# torch.nn.init.xavier_normal_(model.aux.weight)
for param in model.aux.parameters():
    param.requires_grad = True
# for param in model.encoder.transformer.layers[11].parameters():
#     param.requires_grad = True
def init_weights_bias(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=.0, std=1.)
        if module.bias is not None:
            module.bias.data.zero_()
# net = model.encoder.transformer.layers[11]
# net.apply(init_weights_bias)
model.aux.apply(init_weights_bias)

model = model.to(device)

with torch.no_grad():
    i = 2
    sample = audio_dataset[i]
    print(i, sample['audio'].shape, sample['text'])

    waveform = sample['audio']
    emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1).permute(1, 0, 2)

    ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
    target = torch.tensor([label2id(chinese2pinyin(sample['text']))])
    # target =
    print(emissions.shape, target.shape)
    Input_lengths = (emissions.shape[0],)
    Target_lengths = (target.shape[-1],)
    loss = ctc_loss(emissions, target, Input_lengths, Target_lengths)
    
emission = emissions.cpu().detach()
transcript = decoder(emission)
transcript, chinese2pinyin(sample['text']), loss.item()
print('audio_dataset size', len(audio_dataset))


# params = list(model.aux.parameters())+list(model.encoder.transformer.layers[11].parameters())
params = model.aux.parameters()
# optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# optimizer = torch.optim.Adam(model.aux.parameters(), lr=0.01)

"""使用DataLoader按照批次加载"""
from os.path import exists

LOAD_PATH = './checkpoint/model2-no.pt'
if exists(LOAD_PATH):
    print('file',LOAD_PATH,'exist, load checkpoint...')
    checkpoint = torch.load(LOAD_PATH, map_location=device)
    model.aux.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(epoch, loss)

optimizer = torch.optim.SGD(model.aux.parameters(), lr=0.01, momentum=0.9, nesterov=True)
# optimizer = torch.optim.Adam(model.aux.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

batch_size = 128
k_size = model.feature_extractor.conv_layers[0].conv.kernel_size[0]
def collate_wrapper(batch):
    rand_shift = torch.randint(k_size, (batch_size,))
    audio_list = [batch[i]['audio'][:,rand_shift[i]:] for i in range(batch_size)]
    audio_length = torch.tensor([audio.shape[-1] for audio in audio_list])
    max_audio_length = torch.max(audio_length)
    audio_list = torch.cat([
        torch.cat(
        (audio, torch.zeros(max_audio_length-audio.shape[-1]).unsqueeze(0)), -1)
         for audio in audio_list], 0)
    target_list = [label2id(chinese2pinyin(item['text'])) for item in batch]
    target_length = torch.tensor([len(l) for l in target_list])
    max_target_length = torch.max(target_length)
    target_list = torch.cat([
        torch.cat(
        (torch.tensor(l), torch.zeros(max_target_length-len(l))), -1).unsqueeze(0) 
        for l in target_list], 0)
    return {'audio': audio_list.to(device), 'target': target_list.to(device), 'audio_len': audio_length.to(device), 'target_len': target_length.to(device)}

dataloader = DataLoader(audio_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=collate_wrapper)

# iter(dataloader).next()
def test(epoch, k):
    model.eval()
    with torch.no_grad():
        for i in range(k):
            sample = audio_dataset[i]
            save_log(f'e{epoch}.txt', [i, sample['audio'].shape, sample['text']])
            waveform = sample['audio']
            emissions, _ = model(waveform.to(device))
            emissions = torch.log_softmax(emissions, dim=-1)
            emission = emissions[0].cpu().detach()
            transcript = decoder(emission)
            save_log(f'e{epoch}.txt', ['Transcript:', transcript])
            # print('Naive: ', NaiveCTCDecoder(labels)(emission))

def save_checkpoint(EPOCH, LOSS):
    PATH = f"./checkpoint/model_{EPOCH}_{'%.3f' % LOSS}.pt"
    torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.aux.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

def save_log(file_name, log, mode='a', path = './checkpoint/'):
    with open(path+file_name, mode) as f:
        if mode == 'a':
            f.write('\n')
        if type(log) is str:
            f.write(log)
            print(log)
        else:
            log = [str(l) for l in log]
            f.write(' '.join(log))
            print(' '.join(log))

for epoch in range(1):
    model.train()
    current_loss = 0
    for i_batch, sample_batched in enumerate(dataloader):
        batch_loss = 0
        optimizer.zero_grad()
        # for i in range(batch_size):  # Cannot run in batch, only 1 by 1
            
        # Step 1. Prepare Data
        waveform = sample_batched['audio']
        wave_len = sample_batched['audio_len']
        target = sample_batched['target']
        target_len = sample_batched['target_len']

        # Step 2. Run our forward pass
        emissions, emission_len = model(waveform, wave_len)
        # emissions, _ = model(waveform.to(device))
        emissions = torch.nn.functional.log_softmax(emissions, dim=-1).permute(1,0,2)
        # target = torch.tensor([target]).to(device)
        loss = ctc_loss(emissions, target, emission_len, target_len)

        # Step 2. Run our backward pass
        loss.backward()
        
        if loss.item()!=loss.item():
            print('NaN hit!')
            exit()
        current_loss += loss.item()
        batch_loss += loss.item()

        optimizer.step()
        if i_batch % (2000 // batch_size) == 0:
            # print('epoch', epoch, 'lr', scheduler.get_lr(), 'loss', batch_loss/batch_size)
            save_log(f'e{epoch}.txt', ['epoch', epoch, 'lr', scheduler.get_last_lr(), 'loss', batch_loss/batch_size])
            test(epoch, 5)
        
    # scheduler.step()
    save_checkpoint(epoch, current_loss/len(dataloader.dataset))
    test(epoch, 10) # run some sample prediction and see the result





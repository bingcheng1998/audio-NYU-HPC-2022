data_path = './data/zhspeechocean/'
meta_data = data_path + 'metadata.csv'

import pandas as pd

df = pd.read_csv(meta_data, sep='\t')

len(df)

import random
# from IPython.display import Audio

rand_id = random.randint(0, 2400 - 1)

audio_file = df.loc[rand_id]['index']
audio_file = data_path + audio_file

sentence = df.loc[rand_id]['sentence']

print(rand_id, sentence)
# Audio(audio_file)

import os
# from dataclasses import dataclass

# import IPython
# import matplotlib
# import matplotlib.pyplot as plt
import requests
import torch
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

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
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
# labels = {i: s for i, s in enumerate(labels)} # 数字转字母

labels

# indices = torch.argmax(emission, dim=-1)  # [num_seq,]
# torch.unique_consecutive(indices, dim=0)

decoder = GreedyCTCDecoder(labels=labels)
transcript = decoder(emission)
print(transcript)

"""# Transfer Learning in Chinese"""

# ! pip install pypinyin

# import pypinyin
from pypinyin import lazy_pinyin, Style


def chinese2pinyin(text):
    pinyin = lazy_pinyin(text, errors=lambda x: u'-', strict=False)
    if (pinyin[-1]=='-'):
        pinyin = pinyin[:-1]
    pinyin_target = '|'.join(pinyin)
    return pinyin_target.upper()

chinese2pinyin("世界很美好")


def label2id(str):
    return [look_up[i] for i in str]


def id2label(idcs):
    return ''.join([labels[i] for i in idcs])


print(
    label2id(chinese2pinyin("世界很美好")), id2label(label2id(chinese2pinyin("世界很美好")))
)

from torch.utils.data import Dataset, DataLoader

sr = 16000


class AudioDataset(Dataset):

    def __init__(self, meta_data, data_path, sample_rate=16000, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.meta_data = pd.read_csv(meta_data, sep='\t')
        self.root_dir = data_path
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root_dir,
                                  self.meta_data.iloc[idx, 0])
        waveform, sample_rate = torchaudio.load(audio_name)
        waveform = waveform.to(device)

        if sample_rate != sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        audio_content = self.meta_data.iloc[idx, 1]

        sample = {'audio': waveform, 'text': audio_content}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample


audio_dataset = AudioDataset(meta_data, data_path, sample_rate=16000)
audio_dataset[0]

with torch.no_grad():
    i = 1
    sample = audio_dataset[i]
    print(i, sample['audio'].shape, sample['text'])

    waveform = sample['audio']
    emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach().unsqueeze(1)

print(emission.shape, emission[90][0])

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

len(audio_dataset)

print(model.aux)

model = bundle.get_model()
for param in model.parameters():
    param.requires_grad = False
model.aux = torch.nn.Linear(in_features=model.aux.in_features, out_features=model.aux.out_features, bias=True)
# torch.nn.init.xavier_normal(model.aux.weight)
for param in model.aux.parameters():
    param.requires_grad = True
model = model.to(device)

optimizer = torch.optim.SGD(model.aux.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# optimizer = torch.optim.Adam(model.aux.parameters(), lr=0.01)
"""使用DataLoader按照批次加载"""

LOAD_PATH = './checkpoint/model.pt'
print()
checkpoint = torch.load(LOAD_PATH)
model.aux.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
print(epoch, loss)

batch_size = 8
def collate_wrapper(batch):
    audio_list = [item['audio'] for item in batch]
    target_list = [label2id(chinese2pinyin(item['text'])) for item in batch]
    return {'audio': audio_list, 'target': target_list}
dataloader = DataLoader(audio_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=collate_wrapper)

# iter(dataloader).next()

def test(k):
    model.eval()
    with torch.no_grad():
        for i in range(k):
            sample = audio_dataset[i]
            print(i, sample['audio'].shape, sample['text'])
            waveform = sample['audio']
            emissions, _ = model(waveform.to(device))
            emissions = torch.log_softmax(emissions, dim=-1)
            emission = emissions[0].cpu().detach()
            transcript = decoder(emission)
            print('transcript:', transcript, NaiveCTCDecoder(labels)(emission))

def save_checkpoint(EPOCH, LOSS):
    PATH = f"./checkpoint/model_{EPOCH}_{LOSS}.pt"
    torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.aux.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

for epoch in range(40):
    model.train()
    for i_batch, sample_batched in enumerate(dataloader):
        for i in range(batch_size):  # Cannot run in batch, only 1 by 1
            optimizer.zero_grad()
            # Step 1. Prepare Data
            waveform = sample_batched['audio'][i]
            target = sample_batched['target'][i]

            # Step 2. Run our forward pass
            emissions, _ = model(waveform.to(device))
            emissions = torch.nn.functional.log_softmax(emissions, dim=-1).permute(1, 0, 2)
            target = torch.tensor([target]).to(device)
            loss = ctc_loss(emissions, target, (emissions.shape[0],), (target.shape[-1],))

            # Step 2. Run our backward pass
            loss.backward()
            optimizer.step()

        if i_batch % (400 // batch_size) == 0:
            print('epoch', epoch, loss.item())
    scheduler.step()
    save_checkpoint(epoch, loss.item())
    test(3) # run some sample prediction and see the result





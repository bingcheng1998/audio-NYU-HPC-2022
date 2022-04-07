data_path = './data/zhspeechocean/'
meta_data = data_path + 'metadata.csv'

import pandas as pd

df = pd.read_csv(meta_data, sep='\t')

len(df)

import random
from IPython.display import Audio

rand_id = random.randint(0, 2400-1)

audio_file = df.loc[rand_id]['index']
audio_file = data_path+audio_file

sentence = df.loc[rand_id]['sentence']

print(rand_id, sentence)
Audio(audio_file)

"""## Overview

The process of alignment looks like the following.

1. Estimate the frame-wise label probability from audio waveform
2. Generate the trellis matrix which represents the probability of
   labels aligned at time step.
3. Find the most likely path from the trellis matrix.

In this example, we use ``torchaudio``\ ’s ``Wav2Vec2`` model for
acoustic feature extraction.

## Preparation

First we import the necessary packages, and fetch data that we work on.
"""

# %matplotlib inline

import os
from dataclasses import dataclass

import IPython
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio

matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

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

"""## Generate frame-wise label probability

The first step is to generate the label class porbability of each aduio
frame. We can use a Wav2Vec2 model that is trained for ASR. Here we use
:py:func:`torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`.

``torchaudio`` provides easy access to pretrained models with associated
labels.

<div class="alert alert-info"><h4>Note</h4><p>In the subsequent sections, we will compute the probability in
   log-domain to avoid numerical instability. For this purpose, we
   normalize the ``emission`` with :py:func:`torch.log_softmax`.</p></div>



"""

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
with torch.inference_mode():
    waveform, _ = torchaudio.load(SPEECH_FILE)
    emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()

model.aux

emission.shape

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
look_up = {s: i for i, s in enumerate(labels)} # 字母转数字
# labels = {i: s for i, s in enumerate(labels)} # 数字转字母

labels

# indices = torch.argmax(emission, dim=-1)  # [num_seq,]
# torch.unique_consecutive(indices, dim=0)

decoder = GreedyCTCDecoder(labels=labels)
transcript = decoder(emission)
transcript

"""# Transfer Learning in Chinese"""

# ! pip install pypinyin

# import pypinyin
from pypinyin import lazy_pinyin, Style

def chinese2pinyin(text):
  pinyin = lazy_pinyin(text, errors='ignore', strict=False)
  pinyin_target = '|'.join(pinyin)
  return pinyin_target.upper()

chinese2pinyin("世界很美好")

def label2id(str):
  return [look_up[i] for i in str]

def id2label(idcs):
  return ''.join([labels[i] for i in idcs])

label2id(chinese2pinyin("世界很美好")), id2label(label2id(chinese2pinyin("世界很美好")))

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

emission.shape, emission[90][0]

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

model.aux

model = bundle.get_model()
for param in model.parameters():
    param.requires_grad = False
model.aux = torch.nn.Linear(in_features=model.aux.in_features, out_features=model.aux.out_features, bias=True)
# torch.nn.init.xavier_normal(model.aux.weight)
for param in model.aux.parameters():
    param.requires_grad = True
model = model.to(device)

optimizer = torch.optim.SGD(model.aux.parameters(), lr=0.0002, momentum=0.9)

"""使用DataLoader按照批次加载"""

batch_size = 8

def collate_wrapper(batch):
  audio_list = [item['audio'] for item in batch]
  target_list = [label2id(chinese2pinyin(item['text'])) for item in batch]
  return {'audio': audio_list, 'target': target_list}

dataloader = DataLoader(audio_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=collate_wrapper)

# iter(dataloader).next()

model.train()
for epoch in range(20):
  for i_batch, sample_batched in enumerate(dataloader):
    for i in range(batch_size): # Cannot run in batch, only 1 by 1
        optimizer.zero_grad()
        # Step 1. Prepare Data
        waveform = sample_batched['audio'][i]
        target = sample_batched['target'][i]
        
        # Step 2. Run our forward pass
        emissions, _ = model(waveform.to(device))
        emissions = emissions.permute(1, 0, 2).log_softmax(dim=-1).requires_grad_()
        target = torch.tensor([target]).to(device)
        loss = ctc_loss(emissions, target, (emissions.shape[0],), (target.shape[-1],))
        
        # Step 2. Run our backward pass
        loss.backward()
        optimizer.step()

    if i_batch%(400 // batch_size)==0:
          print(epoch, loss.item())

with torch.no_grad():
  i = 1
  sample = audio_dataset[i]
  print(i, sample['audio'].shape, sample['text'])
  
  waveform = sample['audio']
  emissions, _ = model(waveform.to(device))
  emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()
transcript = decoder(emission)
transcript, NaiveCTCDecoder(labels)(emission)

# model.train()
# for epoch in range(10):
#     for i in range(len(audio_dataset)):
#         optimizer.zero_grad()

#         sample = audio_dataset[i]
        
#         # Step 1. Prepare Data
#         waveform = sample['audio']
#         text = sample['text']
        
#         # Step 2. Run our forward pass
#         emissions, _ = model(waveform.to(device))
#         emissions = torch.nn.functional.log_softmax(emissions, dim=-1).permute(1, 0, 2)
#         target = torch.tensor([label2id(chinese2pinyin(sample['text']))])
#         loss = ctc_loss(emissions, target, (emissions.shape[0],), (target.shape[-1],))
        
#         # Step 2. Run our backward pass
#         loss.backward()
#         optimizer.step()
        
#         if i%400==0: # and i == 0:
#           print(epoch, loss.item())

"""### Visualization


"""

print(labels)
plt.imshow(emission.T)
plt.colorbar()
plt.title("Frame-wise class probability")
plt.xlabel("Time")
plt.ylabel("Labels")
plt.show()

"""## Generate alignment probability (trellis)

From the emission matrix, next we generate the trellis which represents
the probability of transcript labels occur at each time frame.

Trellis is 2D matrix with time axis and label axis. The label axis
represents the transcript that we are aligning. In the following, we use
$t$ to denote the index in time axis and $j$ to denote the
index in label axis. $c_j$ represents the label at label index
$j$.

To generate, the probability of time step $t+1$, we look at the
trellis from time step $t$ and emission at time step $t+1$.
There are two path to reach to time step $t+1$ with label
$c_{j+1}$. The first one is the case where the label was
$c_{j+1}$ at $t$ and there was no label change from
$t$ to $t+1$. The other case is where the label was
$c_j$ at $t$ and it transitioned to the next label
$c_{j+1}$ at $t+1$.

The follwoing diagram illustrates this transition.

<img src="https://download.pytorch.org/torchaudio/tutorial-assets/ctc-forward.png">

Since we are looking for the most likely transitions, we take the more
likely path for the value of $k_{(t+1, j+1)}$, that is

$k_{(t+1, j+1)} = max( k_{(t, j)} p(t+1, c_{j+1}), k_{(t, j+1)} p(t+1, repeat) )$

where $k$ represents is trellis matrix, and $p(t, c_j)$
represents the probability of label $c_j$ at time step $t$.
$repeat$ represents the blank token from CTC formulation. (For the
detail of CTC algorithm, please refer to the *Sequence Modeling with CTC*
[`distill.pub <https://distill.pub/2017/ctc/>`__])



"""

transcript = transcript
dictionary = {c: i for i, c in enumerate(labels)}

tokens = [dictionary[c] for c in transcript]
print(list(zip(transcript, tokens)))


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.full((num_frame + 1, num_tokens + 1), -float("inf"))
    trellis[:, 0] = 0
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


trellis = get_trellis(emission, tokens)

"""### Visualization


"""

plt.imshow(trellis[1:, 1:].T, origin="lower")
plt.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
plt.colorbar()
plt.show()

"""In the above visualization, we can see that there is a trace of high
probability crossing the matrix diagonally.

## Find the most likely path (backtracking)

Once the trellis is generated, we will traverse it following the
elements with high probability.

We will start from the last label index with the time step of highest
probability, then, we traverse back in time, picking stay
($c_j \rightarrow c_j$) or transition
($c_j \rightarrow c_{j+1}$), based on the post-transition
probability $k_{t, j} p(t+1, c_{j+1})$ or
$k_{t, j+1} p(t+1, repeat)$.

Transition is done once the label reaches the beginning.

The trellis matrix is used for path-finding, but for the final
probability of each segment, we take the frame-wise probability from
emission matrix.
"""

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]


path = backtrack(trellis, emission, tokens)
print(path)

"""### Visualization


"""

def plot_trellis_with_path(trellis, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    plt.imshow(trellis_with_path[1:, 1:].T, origin="lower")


plot_trellis_with_path(trellis, path)
plt.title("The path found by backtracking")
plt.show()

"""Looking good. Now this path contains repetations for the same labels, so
let’s merge them to make it close to the original transcript.

When merging the multiple path points, we simply take the average
probability for the merged segments.



"""

# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


segments = merge_repeats(path)
for seg in segments:
    print(seg)

"""### Visualization


"""

def plot_trellis_with_segments(trellis, segments, transcript):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))
    ax1.set_title("Path, label and probability for each label")
    ax1.imshow(trellis_with_path.T, origin="lower")
    ax1.set_xticks([])

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start + 0.7, i + 0.3), weight="bold")
            ax1.annotate(f"{seg.score:.2f}", (seg.start - 0.3, i + 4.3))

    ax2.set_title("Label probability with and without repetation")
    xs, hs, ws = [], [], []
    for seg in segments:
        if seg.label != "|":
            xs.append((seg.end + seg.start) / 2 + 0.4)
            hs.append(seg.score)
            ws.append(seg.end - seg.start)
            ax2.annotate(seg.label, (seg.start + 0.8, -0.07), weight="bold")
    ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    xs, hs = [], []
    for p in path:
        label = transcript[p.token_index]
        if label != "|":
            xs.append(p.time_index + 1)
            hs.append(p.score)

    ax2.bar(xs, hs, width=0.5, alpha=0.5)
    ax2.axhline(0, color="black")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(-0.1, 1.1)


plot_trellis_with_segments(trellis, segments, transcript)
plt.tight_layout()
plt.show()

"""Looks good. Now let’s merge the words. The Wav2Vec2 model uses ``'|'``
as the word boundary, so we merge the segments before each occurance of
``'|'``.

Then, finally, we segment the original audio into segmented audio and
listen to them to see if the segmentation is correct.



"""

# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


word_segments = merge_words(segments)
for word in word_segments:
    print(word)

"""### Visualization


"""

def plot_alignments(trellis, segments, word_segments, waveform):
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))

    ax1.imshow(trellis_with_path[1:, 1:].T, origin="lower")
    ax1.set_xticks([])
    ax1.set_yticks([])

    for word in word_segments:
        ax1.axvline(word.start - 0.5)
        ax1.axvline(word.end - 0.5)

    for i, seg in enumerate(segments):
        if seg.label != "|":
            ax1.annotate(seg.label, (seg.start, i + 0.3))
            ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 4), fontsize=8)

    # The original waveform
    ratio = waveform.size(0) / (trellis.size(0) - 1)
    ax2.plot(waveform)
    for word in word_segments:
        x0 = ratio * word.start
        x1 = ratio * word.end
        ax2.axvspan(x0, x1, alpha=0.1, color="red")
        ax2.annotate(f"{word.score:.2f}", (x0, 0.8))

    for seg in segments:
        if seg.label != "|":
            ax2.annotate(seg.label, (seg.start * ratio, 0.9))
    xticks = ax2.get_xticks()
    plt.xticks(xticks, xticks / bundle.sample_rate)
    ax2.set_xlabel("time [second]")
    ax2.set_yticks([])
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlim(0, waveform.size(-1))


plot_alignments(
    trellis,
    segments,
    word_segments,
    waveform[0],
)
plt.show()


# A trick to embed the resulting audio to the generated file.
# `IPython.display.Audio` has to be the last call in a cell,
# and there should be only one call par cell.
def display_segment(i):
    ratio = waveform.size(1) / (trellis.size(0) - 1)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    filename = f"_assets/{i}_{word.label}.wav"
    torchaudio.save(filename, waveform[:, x0:x1], bundle.sample_rate)
    print(f"{word.label} ({word.score:.2f}): {x0 / bundle.sample_rate:.3f} - {x1 / bundle.sample_rate:.3f} sec")
    return IPython.display.Audio(filename)

# Generate the audio for each segment
print(transcript)
IPython.display.Audio(SPEECH_FILE)

display_segment(0)

display_segment(1)

display_segment(2)

display_segment(3)

display_segment(4)

display_segment(5)

display_segment(6)

display_segment(7)

display_segment(8)

"""## Conclusion

In this tutorial, we looked how to use torchaudio’s Wav2Vec2 model to
perform CTC segmentation for forced alignment.



"""
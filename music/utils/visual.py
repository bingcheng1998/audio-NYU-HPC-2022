import os

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio

# matplotlib.rcParams['font.family'] = ['SimHei'] # linux
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS, SimHei'] # macos
matplotlib.rcParams['figure.dpi'] = 200
import librosa

from .helper import initial_table
from .dataset import get_audio, parser_line


def merge_note(text, phoneme, note, note_duration):
    # remove the duplicate items in phoneme, note, and note_duration
    # use text to verify the length
    phoneme = phoneme.copy()
    note = note.copy()
    note_duration = note_duration.copy()
    j = -1
    text+='////////////////////'
    text_with_p = phoneme.copy()
    used_flag = False
    for i in range(len(text_with_p)):
        if text_with_p[i] in ['AP', 'SP']:
            continue
        if j==-1 or phoneme[i] in initial_table or (phoneme[i-1] not in initial_table and phoneme[i] != phoneme[i-1]):
            j+=1
            used_flag = False
        text_with_p[i] = text[j] if used_flag == False else '~'
        used_flag = True
    for i in range(len(phoneme)-1, 0, -1):
        if (note_duration[i] == note_duration[i-1] and phoneme[i-1] in initial_table):
            del note_duration[i]
            del note[i]
            phoneme[i-1]=phoneme[i-1]+phoneme[i]
            del phoneme[i]
            del text_with_p[i]
    return text_with_p, phoneme, note, note_duration


def melspec(waveform, sr=16000):
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )
    melspec = mel_spectrogram(waveform)
    return melspec

def plot_alignment(waveform, text, phoneme, note, note_duration, phoneme_duration, slur_note, save_png=False, sr=16000):
    fontsize = 14
    text_with_p, phoneme_merge, note, note_duration = merge_note(text, phoneme, note, note_duration)
    fig, ax = plt.subplots(3, 1, figsize=(21, 14))
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    [ax1, ax2, ax3] = ax
    # ax1 waveform
    ratio = sr
    ax1.plot(waveform/(max(torch.max(waveform), -torch.min(waveform)))*0.8)
    ax1.set_xlim(0, waveform.size(-1))
    ax1.set_ylim(-1.0, 1.0)
    time_current = 0.
    for i in range(len(phoneme_duration)):
        x0 = ratio * time_current
        time_current += phoneme_duration[i]
        x1 = ratio * time_current
        shift_pos = (phoneme_duration[i]-0.05)*ratio/2 if phoneme_duration[i] > 0.1 else 0.35**ratio
        ax1.axvspan(x0, x1, ymin=0.5, ymax=1, alpha=0.1, color="red")
        ax1.annotate(phoneme[i], (x0+shift_pos, 0.85), color='black', fontsize=fontsize)
    time_current = 0
    for i in range(len(note_duration)):
        x0 = ratio * time_current
        time_current += note_duration[i]
        x1 = ratio * time_current
        ax1.axvspan(x0, x1, ymin=0, ymax=0.5, alpha=0.1, color="blue")
        ax1.annotate(note[i].split('/')[0], (x0+(note_duration[i]-0.05)*ratio/2, -0.85), color='black', fontsize=fontsize)
    # ax2 mel spectrogram
    mel = melspec(waveform)
    ax2.imshow(librosa.power_to_db(mel), origin='lower', aspect='auto', interpolation='none') # interpolation no smooth
    time_precision = 0.5
    for i in range(int((sum(phoneme_duration)/time_precision)//1)):
        x = i*16000.0/512*time_precision
        y = mel.shape[0]
        ax2.annotate(i*time_precision, (x, y-10), color='white', fontsize=fontsize)
    # ax3 information
    ax3.set_ylim(-1.0, 1.0)
    ax3.set_xlim(0, waveform.size(-1))
    y_split = lambda k: [1.0*(i)/k for i in range(k)]
    k = 6
    ys = y_split(k)
    for i in range(k):
        ax3.axhline(y=ys[i]*2-1)
    time_current = 0
    for i in range(len(note_duration)):
        x0 = ratio * time_current
        time_current += note_duration[i]
        x1 = ratio * time_current
        ax3.axvline(x=x1, ymin=ys[1], ymax=ys[4], color="c")
        ax3.annotate(note[i].split('/')[0], (x0+(note_duration[i]-0.05)*ratio/2, 0.1), color='black', fontsize=fontsize)
        ax3.annotate(text_with_p[i], (x0+(note_duration[i]-0.05)*ratio/2, -0.2), color='black', fontsize=fontsize)
        ax3.annotate(phoneme_merge[i], (x0+(note_duration[i]-0.1)*ratio/2, -0.5), color='black', fontsize=fontsize)
    phoneme *2
    time_current = 0
    for i in range(len(phoneme_duration)):
        x0 = ratio * time_current
        time_current += phoneme_duration[i]
        x1 = ratio * time_current
        ax3.axvline(x=x1, ymin=ys[4], ymax=1, color="g")
        shift_pos = (phoneme_duration[i]-0.05)*ratio/2 if phoneme_duration[i] > 0.1 else 0.35**ratio
        ax3.annotate(phoneme[i], (x0+shift_pos, 0.8), color='black', fontsize=fontsize)
        ax3.annotate(slur_note[i], (x0+shift_pos, 0.5), color='black', fontsize=fontsize)
    ax3.annotate(text, ((sum(phoneme_duration)-len(text)/8)*ratio/2, -0.9), color='black', fontsize=fontsize)
    plt.subplots_adjust(wspace =0, hspace =0)
    if save_png: plt.savefig('./img.png', dpi=200, transparent=False)
    plt.show()


def plot_line(line, path):
    id, text, phoneme, note, note_duration, phoneme_duration, slur_note = parser_line(line)
    waveform = get_audio(id, path)
    plot_alignment(waveform[0], text, phoneme, note, note_duration, phoneme_duration, slur_note)
    
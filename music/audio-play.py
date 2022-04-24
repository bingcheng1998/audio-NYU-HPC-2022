import torchaudio
import torch
from utils.dataset import get_audio, parser_line
import pyaudio, struct


gain = 0.9 * 2**15
def get_transcriptions(path):
    with open(path) as f:
        lines = f.read().split('\n')
        return lines
path = './data/segments/'
lines = get_transcriptions(path+'train.txt')


def play(waveform):
    assert len(waveform.shape) == 1
    waveform = (waveform*gain).to(torch.int32).tolist()
    p = pyaudio.PyAudio()
    stream = p.open(
        format = pyaudio.paInt16,  
        channels = 1, 
        rate = 16000,
        input = False, 
        output = True,
        frames_per_buffer = 128)
    binary_data = struct.pack('h' * len(waveform), *waveform)   # 'h' for 16 bits
    stream.write(binary_data)
    stream.stop_stream()
    stream.close()
    p.terminate()

# play(waveform)

id, text, phoneme, note, note_duration, phoneme_duration, slur_note = parser_line(lines[4])
waveform = get_audio(id, path)

def play_phoneme(i, phoneme_duration, waveform):
    assert i < len(phoneme_duration)
    i+=1
    start = int(sum(phoneme_duration[:i-1])*16000) if i > 1 else 0
    end = int(sum(phoneme_duration[:i])*16000)
    phoneme_cut = waveform[start:end]
    play(phoneme_cut)

import time
# >>> time.sleep(3) # Sleep for 3 seconds
for i in range(len(phoneme_duration)):
    print(i, phoneme[i])
    play_phoneme(i, phoneme_duration, waveform[0])
    # time.sleep(1)
    
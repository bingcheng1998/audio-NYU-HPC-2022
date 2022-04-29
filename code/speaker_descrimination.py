import torch
import torchaudio
from model.speaker_encoder import SpeakerEncoder
from torch.nn.utils.rnn import pad_sequence
from os.path import exists
# mean = lambda x: sum(x)/len(x)

sampling_rate = 16000
mel_window_length = 25
mel_window_step = 10
mel_n_channels = 40
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sampling_rate,
    n_fft=int(sampling_rate * mel_window_length / 1000),
    hop_length=int(sampling_rate * mel_window_step / 1000),
    n_mels=mel_n_channels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_decoder():
    model = SpeakerEncoder(40, 256, 256)
    def load_checkpoint(path):
        if exists(path):
            checkpoint = torch.load(path, map_location=device)
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
    LOAD_PATH = './checkpoint/speaker_enc/encoder.pt'
    load_checkpoint(LOAD_PATH)
    model.eval()
    return model

def get_mel_slices(frames_len):
    # 去头去尾中间五分
    cut = 0.15
    n = 5
    content = frames_len*(1-2*cut)
    mel_slices = []
    if content > 160:
        start = frames_len*cut
        hop = int((content-160)/(n-1))
        for i in range(n):
            mel_slices.append(slice(int(start+hop*i),int(start+160+hop*i)))
    else:
        for i in range(n):
            mel_slices.append(slice(int(max(0, frames_len-10-160-10*i)), int(frames_len-10-10*i)))
    return mel_slices

def infer(model, wav):
    frames = mel_transform(wav)
    print(frames.shape)
    mel_slices = get_mel_slices(frames.shape[-1])
    print(mel_slices)
    frames_batch = pad_sequence([frames[:, :, s].permute(0,2,1).squeeze() for s in mel_slices], batch_first=True, padding_value=0.0)
    print(frames_batch.shape)
    partial_embeds = model(frames_batch)
    raw_embed = torch.mean(partial_embeds, dim=0)
    embed = raw_embed / torch.norm(raw_embed, 2)
    return embed

if __name__ == '__main__':
    waveform, sample_rate = torchaudio.load('./audio-temp.wav')
    if sample_rate != sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, sampling_rate)
    model = get_decoder()
    emb = infer(model, waveform)
    print(emb)
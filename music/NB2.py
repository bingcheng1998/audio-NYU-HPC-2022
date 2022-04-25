import matplotlib.pyplot as plt
import time, sys, math
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from utils.display import *
from utils.dsp import *
from models.wavernn import WaveRNN

from utils.dataset import get_audio

def save_log(file_name, log, mode='a', path = './log/n7-'):
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

file = './data/segments/'
sample_rate = 22050
waveform = get_audio(2056002088, file, sr=sample_rate)
# waveform = waveform[:, 20000:30000]
sample = (waveform[0] * (2**15)).int()

notebook_name = '_assets'

coarse_classes, fine_classes = split_signal(sample)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaveRNN().to(device)#.cuda()

from os.path import exists
LOAD_PATH = './checkpoint/1k-no.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_checkpoint(path):
    if exists(path):
        checkpoint = torch.load(path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
load_checkpoint(LOAD_PATH)

coarse_classes, fine_classes = split_signal(sample)

batch_size = 32
coarse_classes = coarse_classes[:len(coarse_classes) // batch_size * batch_size]
fine_classes = fine_classes[:len(fine_classes) // batch_size * batch_size]
coarse_classes = np.reshape(coarse_classes, (batch_size, -1))
fine_classes = np.reshape(fine_classes, (batch_size, -1))

def dump_model(PATH):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)

def train(model, optimizer, num_steps, batch_size, seq_len=960) :
    
    start = time.time()
    running_loss = 0
    
    for step in range(num_steps) :
        
        loss = 0
        hidden = model.init_hidden(batch_size)
        optimizer.zero_grad()
        rand_idx = np.random.randint(0, coarse_classes.shape[1] - seq_len - 1)
        
        for i in range(seq_len) :
            
            j = rand_idx + i
            
            x_coarse = coarse_classes[:, j:j + 1]
            x_fine = fine_classes[:, j:j + 1]
            x_input = np.concatenate([x_coarse, x_fine], axis=1)
            x_input = x_input / 127.5 - 1.
            x_input = torch.FloatTensor(x_input).to(device)#.cuda()
            
            y_coarse = coarse_classes[:, j + 1]
            y_fine = fine_classes[:, j + 1]
            y_coarse = y_coarse.long().to(device)#.cuda()
            y_fine = y_fine.long().to(device)#.cuda()
            
            current_coarse = y_coarse.float() / 127.5 - 1.
            current_coarse = current_coarse.unsqueeze(-1)
            
            out_coarse, out_fine, hidden = model(x_input, hidden, current_coarse)
            
            loss_coarse = F.cross_entropy(out_coarse, y_coarse)
            loss_fine = F.cross_entropy(out_fine, y_fine)
            loss += (loss_coarse + loss_fine)
        
        running_loss += (loss.item() / seq_len)
        loss.backward()
        optimizer.step()
        
        speed = (step + 1) / (time.time() - start)
        
        # stream('Step: %i/%i --- Loss: %.2f --- Speed: %.1f batches/second ',
        #       (step + 1, num_steps, running_loss / (step + 1), speed))  
        if step % 50 == 0:
            save_log(f'e.txt', [f'step: {step + 1}/{num_steps}, loss: {running_loss / (step + 1)}, Speed: {speed}'])
            dump_model('./checkpoint/1k.pt')

optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('start training')
train(model, optimizer, num_steps=2000, batch_size=batch_size)


            
dump_model('./checkpoint/1k.pt')

output, c, f = model.generate(10000)
def save_wav(y, filename, sample_rate) :
    y = np.clip(y, -2**15, 2**15 - 1)
    wavfile.write(filename, sample_rate, y.astype(np.int16))

save_wav(output, f'{notebook_name}/1k_steps.wav', sample_rate)
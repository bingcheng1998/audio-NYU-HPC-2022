import os
from statistics import mean
import torch
import torch.nn as nn
import torchaudio
from pypinyin import lazy_pinyin, Style
from os.path import exists

from utils.textDecoder import GreedyCTCDecoder, NaiveDecoder

from utils.dataset import SpeechOceanDataset, LoaderGenerator
from utils.helper import get_labels

from model.quartznet import QuartzNet
from model.config import quartznet5x5_config

def save_log(file_name, log, mode='a', path = './log/n8-'):
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

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_log(f'e.txt', ['torch:', torch.__version__])
save_log(f'e.txt', ['torchaudio:', torchaudio.__version__])
save_log(f'e.txt', ['device:', device])
save_log(f'e.txt', ['HPC Node:', os.uname()[1]])

def audio_transform(sample, sample_rate):
        audio = sample['audio']
        text = sample['text']
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,\
            n_fft=1024,power=1,hop_length=256,win_length=1024, n_mels=80, \
                f_min=0.0, f_max=8000.0, mel_scale="slaney", norm="slaney")
        def chinese2pinyin(text):
            pinyin = lazy_pinyin(text, strict=True,errors=lambda x: u'')
            pinyin = [i for i in '|'.join(pinyin)]
            return pinyin
        safe_log = lambda x: torch.log(x+2**(-15))
        return {'audio':safe_log(mel_transform(audio)),
                'text': chinese2pinyin(text),
                'chinese': text}
batch_size = 8
test_batch = 8
save_log(f'e.txt', ['Loading Dataset ...'])
dataset = SpeechOceanDataset('./data/zhspeechocean/', transform=audio_transform)
labels = get_labels()
loaderGenerator = LoaderGenerator(labels, k_size=33)
train_set, test_set = dataset.split()
train_loader = loaderGenerator.dataloader(train_set, batch_size=batch_size)
test_loader = loaderGenerator.dataloader(test_set, batch_size=test_batch)

save_log(f'e.txt', ['train_set:', len(train_set), 'test_set:',len(test_set)])
save_log(f'e.txt', ['train batch_size:', batch_size, ', test batch_size', test_batch])

# steps = 10
# for i_batch, sample_batched in enumerate(train_loader):
#     print(sample_batched['audio'].shape, sample_batched['target'].shape)
#     if steps < 0:
#         break
#     steps -= 1


model = QuartzNet(quartznet5x5_config, feat_in = 80, vocab_size=len(labels))

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
ctc_loss = torch.nn.CTCLoss(zero_infinity=True)

decoder = GreedyCTCDecoder(labels=labels)

save_log(f'e.txt', ['Init Model ...'])
model = model.to(device)

def test_decoder(epoch, k):
    model.eval()
    with torch.no_grad():
        for i in range(k):
            sample = test_set[i]
            save_log(f'e{epoch}.txt', [i, sample['audio'].shape, sample['chinese']])
            waveform = sample['audio']
            emissions, _ = model(waveform.to(device))
            print('emissions', emissions.shape)
            emissions = torch.log_softmax(emissions, dim=-2)
            emission = emissions[0].cpu().detach().T
            # print('emission.shape', emission.shape)
            # print('emission', emission)
            transcript = decoder(emission)
            save_log(f'e{epoch}.txt', ['Transcript:', transcript])

test_decoder('', 5)

NUM_EPOCHS=5

def train(epoch=1):
    train_loss_q = []
    test_loss_q = []
    for epoch in range(0, epoch):
        model.train()
        batch_train_loss = []
        for i_batch, sample_batched in enumerate(train_loader):

            # Step 1. Prepare Data
            waveform = sample_batched['audio']
            wave_len = sample_batched['audio_len']
            target = sample_batched['target']
            target_len = sample_batched['target_len']

            # Step 2. Run our forward pass
            emissions, emission_len = model(waveform, wave_len)
            emissions = torch.log_softmax(emissions, dim=-2).permute(2,0,1)
            print(emissions.shape, target.shape, emission_len.shape, target_len.shape)
            # print(emissions[0], target[0], wave_len[0], target_len[0])
            loss = ctc_loss(emissions, target, emission_len, target_len)

            # Step 3. Run our backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item()!=loss.item(): # if loss == NaN, break
                print('NaN hit!')
                exit()
            
            batch_train_loss.append(loss.item())

            if i_batch % (1000 // batch_size) == 0: # log about each 1000 data
                # test_loss = test()
                test_loss = 0
                train_loss = mean(batch_train_loss)
                test_loss_q.append(test_loss)
                train_loss_q.append(train_loss)
                save_log(f'e{epoch}.txt', ['🟣 epoch', epoch, 'data', i_batch*batch_size, 
                    'lr', scheduler.get_last_lr(), 
                    'train_loss', train_loss, 'test_loss', test_loss])
                # save_temp(epoch, test_loss) # save temp checkpoint
                test_decoder(epoch, 5)
            
        scheduler.step()
        # save_checkpoint(epoch, mean(test_loss_q))
        save_log(f'e{epoch}.txt', ['============= Final Test ============='])
        test_decoder(epoch, 10) # run some sample prediction and see the result

train(NUM_EPOCHS)

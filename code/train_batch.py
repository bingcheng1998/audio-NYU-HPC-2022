import os
import random
from statistics import mean
import requests
import torch
import torch.nn as nn
import torchaudio
from pypinyin import lazy_pinyin, Style
from os.path import exists

from utils.textDecoder import GreedyCTCDecoder, NaiveDecoder
from utils.dataset import AudioDataset, LoaderGenerator
from utils.helper import get_labels

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('torch:\t', torch.__version__)
print('torchaudio:', torchaudio.__version__)
print('device:\t', device)

LOAD_PATH = './checkpoint/model_no.pt' # checkpoint used if exist
batch_size = 64
bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
wave2vec_model = bundle.get_model()
labels = get_labels()
k_size = wave2vec_model.feature_extractor.conv_layers[0].conv.kernel_size[0] # kernel size for audio encoder
mean = lambda x: sum(x)/len(x)

def chinese2pinyin(text):
    initials = lazy_pinyin(text, strict=True, style=Style.INITIALS, errors=lambda x: u'')
    finals = lazy_pinyin(text, strict=True, style=Style.FINALS, errors=lambda x: u'')
    pinyin = ''
    for i in range(len(finals)):
        pinyin+='|'
        if (initials[i] == '-'):
            continue
        pinyin+=initials[i]
        pinyin+=finals[i]
        if finals[i] == '':
            pinyin+='n'
    if pinyin[-1] == '|':
        pinyin = pinyin[:-1]
    return pinyin[1:].lower().replace('w','u')

# def chinese2pinyin(text):
#     pinyin = lazy_pinyin(text, strict=True,errors=lambda x: u'')
#     pinyin = [i for i in '|'.join(pinyin)]
#     return ['|']+pinyin

dataset = AudioDataset('./data/ST-CMDS-20170001_1-OS/')
train_set, test_set = dataset.split([8, 2])
loaderGenerator = LoaderGenerator(labels, chinese2pinyin, k_size)
train_loader = loaderGenerator.dataloader(train_set, batch_size)
test_loader = loaderGenerator.dataloader(test_set, batch_size)

decoder = GreedyCTCDecoder(labels=labels)

class ChineseStt(torch.nn.Module):
    def __init__(self, wave2vec_model, out_features):
        super(ChineseStt, self).__init__()
        self.feature_extractor = wave2vec_model.feature_extractor
        self.encoder = wave2vec_model.encoder
        in_features = wave2vec_model.encoder.transformer.layers[-1].final_layer_norm.normalized_shape[0]
        self.aux = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x, lengths=None):
        x, lengths = self.feature_extractor(x, lengths)
        x = self.encoder(x, lengths)
        x = self.aux(x)
        return x, lengths

model = ChineseStt(wave2vec_model, len(labels))
for param in model.feature_extractor.parameters():
    param.requires_grad = False
torch.nn.init.xavier_normal_(model.aux.weight)
for param in model.encoder.parameters():
    param.requires_grad = True
model = model.to(device)

if exists(LOAD_PATH):
    print('file',LOAD_PATH,'exist, load checkpoint...')
    checkpoint = torch.load(LOAD_PATH, map_location=device)
    model.aux.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(epoch, loss)

# params = list(model.aux.parameters())+list(model.encoder.transformer.layers[11].parameters())
params = model.aux.parameters()
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
ctc_loss = torch.nn.CTCLoss(zero_infinity=True)

def test_decoder(epoch, k):
    model.eval()
    with torch.no_grad():
        for i in range(k):
            sample = test_set[i]
            save_log(f'e{epoch}.txt', [i, sample['audio'].shape, sample['text']])
            waveform = sample['audio']
            emissions, _ = model(waveform.to(device))
            emissions = torch.log_softmax(emissions, dim=-1)
            emission = emissions[0].cpu().detach()
            transcript = decoder(emission)
            save_log(f'e{epoch}.txt', ['Transcript:', transcript])

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

def test():
    model.eval()
    losses = []
    for sample_batched in test_loader:

        # Step 1. Prepare Data
        waveform = sample_batched['audio']
        wave_len = sample_batched['audio_len']
        target = sample_batched['target']
        target_len = sample_batched['target_len']

        # Step 2. Run our forward pass
        emissions, emission_len = model(waveform.to(device), wave_len.to(device))
        emissions = torch.log_softmax(emissions, dim=-1).permute(1,0,2)
        loss = ctc_loss(emissions, target, emission_len, target_len)

        # Step 2. Run our backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item()!=loss.item(): # if loss == NaN, break
            print('NaN hit!')
            exit()

        losses.append(loss.item())

    return mean(losses)

def train(epoch=1):
    train_loss_q = []
    test_loss_q = []
    for epoch in range(epoch):
        model.train()
        batch_train_loss = []
        for i_batch, sample_batched in enumerate(train_loader):

            # Step 1. Prepare Data
            waveform = sample_batched['audio']
            wave_len = sample_batched['audio_len']
            target = sample_batched['target']
            target_len = sample_batched['target_len']

            # Step 2. Run our forward pass
            emissions, emission_len = model(waveform.to(device), wave_len.to(device))
            emissions = torch.log_softmax(emissions, dim=-1).permute(1,0,2)
            loss = ctc_loss(emissions, target, emission_len, target_len)

            # Step 2. Run our backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item()!=loss.item(): # if loss == NaN, break
                print('NaN hit!')
                exit()
            
            batch_train_loss.append(loss.item())

            if i_batch % (2000 // batch_size) == 0:
                test_loss = test()
                train_loss = mean(batch_train_loss)
                test_loss_q.append(test_loss)
                train_loss_q.append(train_loss)
                save_log(f'e{epoch}.txt', ['🟣 epoch', epoch, 'data', i_batch*batch_size, 'lr', scheduler.get_last_lr(), 
                    'train_loss', train_loss, 'test_loss', test_loss])
                test_decoder(epoch, 5)
            
        scheduler.step()
        save_checkpoint(epoch, mean(test_loss))
        save_log(f'e{epoch}.txt', ['============= 最终测试 =============='])
        test_decoder(epoch, 10) # run some sample prediction and see the result

train()


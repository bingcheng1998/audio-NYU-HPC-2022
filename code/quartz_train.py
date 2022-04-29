import os
# from statistics import mean
import torch
import torch.nn as nn
import torchaudio
from pypinyin import lazy_pinyin, Style
from os.path import exists

from utils.textDecoder import GreedyCTCDecoder, NaiveDecoder

from utils.dataset import SpeechOceanDataset, LoaderGenerator, STCMDSDataset, AiShellDataset
# from utils.helper import get_labels

# from model.quartznet import QuartzNet
from model.quartz2 import QuartzNet
# from model.quartznet import QuartzNet
from utils.chinese2pinyin2 import chinese2pinyin, get_labels

mean = lambda x: sum(x)/len(x)


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

LOAD_PATH = './checkpoint/quartz/model-temp-no.pt'
# LOAD_PATH = './checkpoint/quartz/epoch_5_2_new_data_0.pt'
N_MELS = 80
NUM_EPOCHS=20
# torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_log(f'e.txt', ['torch:', torch.__version__])
save_log(f'e.txt', ['torchaudio:', torchaudio.__version__])
save_log(f'e.txt', ['device:', device])
save_log(f'e.txt', ['HPC Node:', os.uname()[1]])

def audio_transform(sample, sample_rate):
        audio = sample['audio']
        text = sample['text']
        mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,\
            n_fft=1024,power=1,hop_length=256,win_length=1024, n_mels=N_MELS, \
                f_min=0.0, f_max=8000.0, mel_scale="slaney", norm="slaney")

        # def chinese2pinyin(text):
        #     pinyin = lazy_pinyin(text, strict=True,errors=lambda x: u'')
        #     pinyin = [i for i in '|'.join(pinyin)]
        #     return pinyin
        # safe_log = lambda x: torch.log(x+2**(-15))
        return {'audio':mel_transform(audio),
                'text': chinese2pinyin(text),
                'chinese': text}

save_log(f'e.txt', ['Loading Dataset ...'])
# dataset = SpeechOceanDataset('./data/zhspeechocean/', transform=audio_transform)
# dataset = AiShellDataset('./data/data_aishell/', transform=audio_transform)
dataset = STCMDSDataset('./data/ST-CMDS-20170001_1-OS/', transform=audio_transform)
labels = get_labels()
loaderGenerator = LoaderGenerator(labels, k_size=33)
train_set, test_set = dataset.split()
# batch_size = int(train_set.dataset.batch_size*3) # tain batch size
batch_size = 37
test_batch = batch_size
train_loader = loaderGenerator.dataloader(train_set, batch_size=batch_size)
test_loader = loaderGenerator.dataloader(test_set, batch_size=test_batch)
safe_log = lambda x: torch.log(x+2**(-15))

save_log(f'e.txt', ['train_set:', len(train_set), 'test_set:',len(test_set)])
save_log(f'e.txt', ['train batch_size:', batch_size, ', test batch_size', test_batch])


model = QuartzNet(n_mels = N_MELS, num_classes=len(labels))

# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
decoder = GreedyCTCDecoder(labels=labels)
# decoder = NaiveDecoder(labels=labels)

save_log(f'e.txt', ['Init Model ...'])
model = model.to(device)
initial_epoch = 0
def load_checkpoint(path):
    if exists(path):
        save_log(f'e.txt', ['file',path,'exist, load checkpoint...'])
        checkpoint = torch.load(path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        loss = 0
        save_log(f'e.txt', ['initial_epoch:', initial_epoch, 'loss:', loss])
load_checkpoint(LOAD_PATH)

# params = list(model.blocks.parameters())\
#         +list(model.c2.parameters())\
#             +list(model.c3.parameters())
#             # +list(model.c1.parameters())
# for param in params:
#     param.requires_grad = False

# model.classify = nn.Conv1d(1024, len(labels), kernel_size=1, bias=True).to(device)
# torch.nn.init.xavier_normal_(model.classify.weight, gain=nn.init.calculate_gain('relu'))
# for param in model.parameters():
#     param.requires_grad = True
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.2)

def test_decoder(epoch, k):
    model.eval()
    with torch.no_grad():
        for i in range(k):
            sample = test_set[i]
            save_log(f'e{epoch}.txt', [i, sample['audio'].shape, sample['chinese']])
            waveform = sample['audio']
            waveform = safe_log(waveform)
            emissions, _ = model(waveform.to(device))
            # print('emissions', emissions.shape)
            emissions = torch.log_softmax(emissions, dim=-2)
            emission = emissions[0].cpu().detach().T
            # print('emission.shape', emission.shape)
            # print('emission', emission)
            transcript = decoder(emission)
            save_log(f'e{epoch}.txt', ['Transcript:', transcript])

test_decoder('', 5)

def dump_model(EPOCH, LOSS, PATH):
    torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

def save_temp(EPOCH, LOSS):
    PATH = f"./checkpoint/quartz/model-temp.pt"
    dump_model(EPOCH, LOSS, PATH)
    
def save_checkpoint(EPOCH, LOSS):
    PATH = f"./checkpoint/model_{EPOCH}_{'%.3f' % LOSS}.pt"
    dump_model(EPOCH, LOSS, PATH)

def test():
    model.eval()
    losses = []
    with torch.no_grad():
        for sample_batched in test_loader:
            # Step 1. Prepare Data
            waveform = sample_batched['audio']
            wave_len = sample_batched['audio_len']
            target = sample_batched['target']
            target_len = sample_batched['target_len']
            waveform = safe_log(waveform)
            # Step 2. Run our forward pass
            emissions, emission_len = model(waveform, wave_len)
            emissions = torch.log_softmax(emissions, dim=-2).permute(2,0,1)
            loss = ctc_loss(emissions, target, emission_len, target_len)
            losses.append(loss.item())
    return mean(losses)

save_log(f'e.txt', ['initial test loss:', test()])

f_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
t_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
# torch.autograd.set_detect_anomaly(True)
def train(epoch=1):
    train_loss_q = []
    test_loss_q = []
    for epoch in range(initial_epoch, epoch):
        
        batch_train_loss = []
        for i_batch, sample_batched in enumerate(train_loader):
            model.train()
            # Step 1. Prepare Data
            waveform = sample_batched['audio']
            wave_len = sample_batched['audio_len']
            target = sample_batched['target']
            target_len = sample_batched['target_len']
            # waveform = t_mask(waveform)
            # waveform = f_mask(waveform)
            waveform = safe_log(waveform)
            # Step 2. Run our forward pass
            emissions, emission_len = model(waveform, wave_len)
            emissions = torch.log_softmax(emissions, dim=-2).permute(2,0,1)
            # print(emissions.shape, target.shape, emission_len.shape, target_len.shape)
            # print(emissions[0], target[0], wave_len[0], target_len[0])
            loss = ctc_loss(emissions, target, emission_len, target_len)

            # Step 3. Run our backward pass
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
            optimizer.step()

            if loss.item()!=loss.item(): # if loss == NaN, break
                print('NaN hit!')
                exit()
            
            batch_train_loss.append(loss.item())

            if i_batch % (1000 // batch_size) == 0: # log about each 1000 data
                test_loss = test()
                # test_loss = 0
                train_loss = mean(batch_train_loss)
                test_loss_q.append(test_loss)
                train_loss_q.append(train_loss)
                save_log(f'e{epoch}.txt', ['🟣 epoch', epoch, 'data', i_batch*batch_size, 
                    'lr', scheduler.get_last_lr(), 
                    'train_loss', train_loss, 'test_loss', test_loss])
                save_temp(epoch, test_loss) # save temp checkpoint
                test_decoder(epoch, 5)
            
        scheduler.step()
        # save_checkpoint(epoch, mean(test_loss_q))
        save_log(f'e{epoch}.txt', ['============= Final Test ============='])
        test_decoder(epoch, 10) # run some sample prediction and see the result

train(NUM_EPOCHS)

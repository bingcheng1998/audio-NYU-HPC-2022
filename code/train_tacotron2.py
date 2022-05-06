import os

import torch
import torchaudio
from pypinyin import lazy_pinyin
from torch.nn.utils.rnn import pad_sequence

from utils.dataset import AiShell3PersonDataset, RawLoaderGenerator, AiShell3Dataset, MelLoaderGenerator


def save_log(file_name, log, mode='a', path = './log/n3-'):
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

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
processor = bundle.get_text_processor()
# tacotron2 = bundle.get_tacotron2().to(device)

from torchaudio.pipelines._tts.utils import _get_chars

labels = _get_chars() + ('1', '2', '3', '4', '5') # 音调，5是轻声

def raw_audio_transform(sample, sample_rate=None):
        audio = sample['audio']
        audio = torchaudio.functional.vad(audio, sample_rate, trigger_level=5)
        audio = audio / torch.abs(audio).max()*0.15
        text = sample['text']
        text = text.split(' ')
        pinyin = [text[i] for i in range(len(text)) if i%2==1]
        pinyin = ' '.join(pinyin) # 使用空格分离单字
        chinese = [text[i] for i in range(len(text)) if i%2==0]
        return {'audio':audio,
                'text': pinyin,
                'chinese': chinese}
sample_rate = 16000               
# dataset = AiShell3PersonDataset('/scratch/bh2283/data/data_aishell3/train/', transform=raw_audio_transform, \
#         person_id='SSB0011', sample_rate=sample_rate)
dataset = AiShell3Dataset('/scratch/bh2283/data/data_aishell3/train/', transform=raw_audio_transform, sample_rate=sample_rate)

# loaderGenerator = RawLoaderGenerator(labels, k_size=5, num_workers=1)
loaderGenerator = MelLoaderGenerator(labels, k_size=5, num_workers=1, sample_rate=sample_rate)
batch_size = 128
train_set, test_set = dataset.split([1,0])
train_loader = loaderGenerator.dataloader(train_set, batch_size=batch_size)

mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,\
            n_fft=1024,power=1,hop_length=256,win_length=1024, n_mels=80, \
                f_min=0.0, f_max=8000.0, mel_scale="slaney", norm="slaney").to(device)
safe_log = lambda x: torch.log(x+2**(-15))

# my_tacotron2 = bundle.get_tacotron2()
# new_embedding = torch.nn.Embedding(len(labels), tacotron2.embedding.embedding_dim)
# new_embedding.weight[:tacotron2.embedding.num_embeddings, :].data=tacotron2.embedding.weight.data
# my_tacotron2.embedding=new_embedding
from model.MyTacotron2 import MyTacotron2
model = MyTacotron2(labels).to(device)
# model = my_tacotron2.to(device)

params = model.parameters()
# optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(params, lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
initial_epoch = 0
mse_loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCELoss()
cos_loss = torch.nn.CosineEmbeddingLoss()
mean = lambda x: sum(x)/len(x)

def dump_model(EPOCH, LOSS, PATH):
    torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

def save_temp(EPOCH, LOSS):
    PATH = f"./checkpoint/tacotron2/model_temp.pt"
    dump_model(EPOCH, LOSS, PATH)
    
def save_checkpoint(EPOCH, LOSS):
    PATH = f"./checkpoint/tacotron2/model_{EPOCH}_{'%.3f' % LOSS}.pt"
    dump_model(EPOCH, LOSS, PATH)

def train(epoch=1):
    train_loss_q = []
    test_loss_q = []
    for epoch in range(initial_epoch, epoch):
        batch_train_loss = []
        for i_batch, sample_batched in enumerate(train_loader):
            model.train()
            # Step 1. Prepare Data
            # audio = sample_batched['audio'].to(device)
            # audio_len = sample_batched['audio_len']
            tokens = sample_batched['target'].to(device)
            tokens_len = sample_batched['target_len'].to(device)
            mels_tensor = sample_batched['mel'].to(device) # [bs, mel_bins, L]
            mel_length = sample_batched['mel_len'].to(device)

            # Step 2. Run our forward pass
            # mels_list = [safe_log(mel_transform(audio[i][:audio_len[i]])).transpose(0,1) for i in range(len(audio_len))]
            # mel_length = torch.tensor([mel.shape[-2] for mel in mels_list]).to(device)
            # mels_tensor = pad_sequence(mels_list, batch_first=True, padding_value=torch.log(torch.tensor(2**(-15)))).permute(0,2,1)

            speaker_emb = model.speaker_encoder(mels_tensor.transpose(1,2), mel_length)

            org_mel, pos_mel, stop_token, _ = model.forward(tokens, tokens_len, mels_tensor, mel_length, speaker_emb)
            loss1 = mse_loss(mels_tensor, org_mel)
            loss1 += mse_loss(mels_tensor, pos_mel)

            reconstruct_speaker_emb = model.speaker_encoder(pos_mel.transpose(1,2), mel_length)
            loss2 = cos_loss(speaker_emb, reconstruct_speaker_emb, torch.ones_like(tokens_len))

            true_stop_token = torch.zeros(stop_token.shape).to(device)
            for i in range(true_stop_token.shape[0]):
                true_stop_token[i][mel_length[i]:]+=1.0
            loss3 = bce_loss(torch.sigmoid(stop_token), true_stop_token)
            
            # Step 3. Run our backward pass
            optimizer.zero_grad()
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()

            if loss.item()!=loss.item(): # if loss == NaN, break
                print('NaN hit!')
                exit()
            
            batch_train_loss.append(loss.item())

            if i_batch % (500 // batch_size) == 0: # log about each 1000 data
                # test_loss = test()
                test_loss = 0
                train_loss = mean(batch_train_loss)
                test_loss_q.append(test_loss)
                train_loss_q.append(train_loss)
                save_log(f'e{epoch}.txt', ['🟣 epoch', epoch, 'data', i_batch*batch_size, 
                    'lr', scheduler.get_last_lr(), 
                    'train_loss', train_loss, 'test_loss', test_loss])
                save_temp(epoch, test_loss) # save temp checkpoint
                # test_decoder(epoch, 5)
            
        # scheduler.step()
        save_checkpoint(epoch, mean(test_loss_q))
        save_log(f'e{epoch}.txt', ['============= Final Test ============='])
        # test_decoder(epoch, 10) # run some sample prediction and see the result

train(30)

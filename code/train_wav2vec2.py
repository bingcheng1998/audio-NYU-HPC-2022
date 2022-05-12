import os
from statistics import mean
import torch
import torch.nn as nn
import torchaudio
from pypinyin import lazy_pinyin, Style
from os.path import exists

from utils.textDecoder import GreedyCTCDecoder, NaiveDecoder
from utils.dataset import *
# from utils.helper import get_labels

def save_log(file_name, log, mode='a', path = './log/n0-'):
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

NUM_EPOCHS = 5
LOAD_PATH = './checkpoint/wav2vec/model_no.pt' # checkpoint used if exist
# bundle = torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_EN
# wave2vec_model = bundle.get_model()
# labels = get_labels()
# k_size = wave2vec_model.feature_extractor.conv_layers[0].conv.kernel_size[0] 
# kernel size for audio encoder, will be used in audio augmentation

mean = lambda x: sum(x)/len(x)

from utils.helper import get_alphabet_labels, get_phoneme_labels, get_tone_labels, get_pitch_labels

alphabet_labels = get_alphabet_labels()
phoneme_labels = get_phoneme_labels()
tone_labels = get_tone_labels()
alphabet_look_up = {s: i for i, s in enumerate(alphabet_labels)} # label转数字
phoneme_look_up = {s: i for i, s in enumerate(phoneme_labels)}
tone_look_up = {s: i for i, s in enumerate(tone_labels)}

alphabet_decoder = GreedyCTCDecoder(labels=alphabet_labels)
phoneme_decoder = GreedyCTCDecoder(labels=phoneme_labels)
tone_decoder = GreedyCTCDecoder(labels=tone_labels)

def chinese2alphabet(chinese):
    pinyin = lazy_pinyin(chinese, strict=True,errors=lambda x: u'-')
    pinyin = [i for i in '|'.join(pinyin)]
    return pinyin
def chinese2phoneme(chinese):
    intitials = lazy_pinyin(chinese, strict=False, style=Style.INITIALS, errors=lambda x: u'-')
    finals = lazy_pinyin(chinese, strict=True, style=Style.FINALS, errors=lambda x: u'-')
    result = []
    for i in range(len(intitials)):
        result += ['|']
        if intitials[i] != '':
            result += [intitials[i]]
        elif finals[i] == '':
            result += ['n']
        if finals[i] != '':
            result += [finals[i]]
    return result[1:]
def chinese2tone(chinese):
    pinyin = lazy_pinyin(chinese, strict=True, style=Style.TONE3, neutral_tone_with_five=True, errors=lambda x: u'-')
    tone = [i[-1] for i in pinyin]
    return [i for i in  '|'.join(tone)]

labels_list = [alphabet_labels, phoneme_labels, tone_labels]
translators_list = [chinese2alphabet, chinese2phoneme, chinese2tone]
decoders = [alphabet_decoder, phoneme_decoder, tone_decoder]

save_log(f'e.txt', ['Loading Dataset ...'])

class MultiTaskRawLoaderGenerator:
    def __init__(self, 
        labels_list:list, 
        translators_list: list,
        k_size=0, 
        num_workers=0,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> None:
        self.k_size = k_size
        self.labels_list = labels_list
        self.look_up_list = [{s: i for i, s in enumerate(labels)} for labels in labels_list]
        self.device = device
        self.num_workers = num_workers

    def label2id(self, label_set:int, str):
        return [self.look_up_list[label_set][i] for i in str]

    def id2label(self, label_set:int, idcs):
        return ''.join([self.labels_list[label_set][i] for i in idcs])

    def batch_filter(self, batch:list):
        # remove all audio with tag if audio length > threshold
        for i in range(len(batch)-1, -1, -1):
            if batch[i]['audio'].shape[-1] > self.threshold: # 256 is the hop_length of fft
                del batch[i]
        return batch

    def collate_wrapper(self, batch:list): # RAW
        batch = self.batch_filter(batch)
        bs = len(batch)
        rand_shift = torch.randint(self.k_size, (bs,))
        audio_list = [batch[i]['audio'][:,rand_shift[i]:] for i in range(bs)]
        audio_length = [audio.shape[-1] for audio in audio_list]
        audio_length = torch.tensor(audio_length)
        max_audio_length = torch.max(audio_length)
        audio_list = torch.cat([
            torch.cat(
            (audio, torch.zeros(max_audio_length-audio.shape[-1]).unsqueeze(0)), -1)
            for audio in audio_list], 0)

        all_target_list = []
        all_target_length = []
        for label_set in range(len(self.labels_list)):
            target_list = [self.label2id(label_set, translators_list[label_set](item['chinese'])) for item in batch]
            target_length = [len(l) for l in target_list]
            target_length = torch.tensor(target_length)
            max_target_length = torch.max(target_length)
            target_list = torch.cat([
                torch.cat(
                (torch.tensor(l), torch.zeros([max_target_length-len(l)], dtype=torch.int)), -1).unsqueeze(0) 
                for l in target_list], 0)
            all_target_list.append(target_list)
            all_target_length.append(target_length)
        return {'audio': audio_list, 'audio_len': audio_length, 
                'target': all_target_list, 'target_len': all_target_length}

    def dataloader(self, audioDataset, batch_size, shuffle=True):
        # k_size is the kernel size for the encoder, for data augmentation
        self.threshold = audioDataset.dataset.threshold
        return DataLoader(audioDataset, batch_size,
                            shuffle, num_workers=self.num_workers, collate_fn=self.collate_wrapper)

def raw_audio_transform(sample, sample_rate=None):
        audio = sample['audio']
        audio = audio / torch.abs(audio).max()*0.15
        sample['audio'] = audio
        sample['chinese'] = sample['text']
        return sample

dataset = PrimeWordsDataset('/scratch/bh2283/data/primewords_md_2018_set1/', transform=raw_audio_transform)

from model.wav2vec2 import Wav2Vec2Builder
labels_sizes = [len(labels) for labels in labels_list]
builder = Wav2Vec2Builder(torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_EN, labels_sizes)
k_size = builder.kernel_size
train_set, test_set = dataset.split()
batch_size = train_set.dataset.batch_size//2 # tain batch size
# batch_size = 16
test_batch = batch_size//4 # test batch size, keep bs small to save memory
loaderGenerator = MultiTaskRawLoaderGenerator(labels_list, translators_list, k_size)
train_loader = loaderGenerator.dataloader(train_set, batch_size)
test_loader = loaderGenerator.dataloader(test_set, test_batch, shuffle=False)
save_log(f'e.txt', ['train_set:', len(train_set), 'test_set:',len(test_set)])
save_log(f'e.txt', ['train batch_size:', batch_size, ', test batch_size', test_batch])

# decoder = GreedyCTCDecoder(labels=labels)

save_log(f'e.txt', ['Init Model ...'])

from model.wav2vec2 import Wav2Vec2Builder

model = builder.get_model()
save_log(f'e.txt', ['k_size:', builder.kernel_size])

# for param in model.feature_extractor.parameters():
#     param.requires_grad = False
# torch.nn.init.xavier_normal_(model.aux.weight)
# for param in model.encoder.parameters():
#     param.requires_grad = True
# params = list(model.aux.parameters())+list(model.encoder.parameters())
params = []
params += list(model.encoder.parameters())
for i in range(len(model.aux)):
    model.aux[i] = model.aux[i].to(device)
    params += list(model.aux[i].parameters())
# params = model.aux.parameters()
model = model.to(device)

optimizer = torch.optim.Adam(params, lr=0.001)
ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
initial_epoch = 0

save_log(f'e.txt', ['Loading Checkpoint ...'])
def load_checkpoint(path):
    if exists(path):
        print('file',path,'exist, load checkpoint...')
        checkpoint = torch.load(path, map_location=device)
        if 'model_aux_dict' in checkpoint:
            for i in range(len(model.aux)):
                model.aux[i].load_state_dict(checkpoint['model_aux_dict'][i])
        if 'model_encoder_dict' in checkpoint:
            model.encoder.load_state_dict(checkpoint['model_encoder_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(epoch, loss)
load_checkpoint(LOAD_PATH)



def test_decoder(epoch, k):
    model.eval()
    with torch.no_grad():
        for i in range(k):
            sample = test_set[i]
            print(i, sample['audio'].shape, sample['chinese'])
            waveform = sample['audio']
            emissions, _ = model(waveform.to(device))
            emissions = torch.log_softmax(emissions[0], dim=-1)
            emission = emissions[0].cpu().detach()
            transcript = decoders[0](emission)
            save_log(f'e{epoch}.txt', ['Transcript:', transcript])

test_decoder('', 5)

def dump_model(EPOCH, LOSS, PATH):
    torch.save({
            'epoch': EPOCH,
            'model_aux_dict': [model.aux[i].state_dict() for i in range(len(model.aux))],
            'model_encoder_dict': model.encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

def save_temp(EPOCH, LOSS):
    PATH = f"./checkpoint/wav2vec/model_temp.pt"
    dump_model(EPOCH, LOSS, PATH)
    
def save_checkpoint(EPOCH, LOSS):
    PATH = f"./checkpoint/wav2vec/model_{EPOCH}_{'%.3f' % LOSS}.pt"
    dump_model(EPOCH, LOSS, PATH)

def test():
    model.eval()
    losses = []
    with torch.no_grad():
        for sample_batched in test_loader:
            # Step 1. Prepare Data
            waveform = sample_batched['audio'].to(device)
            wave_len = sample_batched['audio_len'].to(device)
            target = [sample_batched['target'][i].to(device) for i in range(len(sample_batched['target']))]
            target_len = [sample_batched['target_len'][i].to(device) for i in range(len(sample_batched['target_len']))]
            al_t, ph_t, tn_t = target[0], target[1], target[2]
            al_l, ph_l, tn_l = target_len[0], target_len[1], target_len[2]
            # Step 2. Run our forward pass
            emissions, emission_len = model(waveform, wave_len)
            emissions = [torch.log_softmax(emission, dim=-1).permute(1,0,2) for emission in emissions]
            al, ph, tn = emissions
            al_loss = ctc_loss(al, al_t, emission_len, al_l)
            ph_loss = ctc_loss(ph, ph_t, emission_len, ph_l)
            tn_loss = ctc_loss(tn, tn_t, emission_len, tn_l)
            loss = al_loss + ph_loss + tn_loss
            losses.append(loss.item())
    return mean(losses)

# save_log(f'e.txt', ['initial test loss:', test()])


save_log(f'e.txt', ['Start training ...'])
def train(epoch=1):
    train_loss_q = []
    test_loss_q = []
    for epoch in range(initial_epoch, epoch):
        
        batch_train_loss = []
        for i_batch, sample_batched in enumerate(train_loader):
            model.train()
            # Step 1. Prepare Data
            waveform = sample_batched['audio'].to(device)
            wave_len = sample_batched['audio_len'].to(device)
            target = [sample_batched['target'][i].to(device) for i in range(len(sample_batched['target']))]
            target_len = [sample_batched['target_len'][i].to(device) for i in range(len(sample_batched['target_len']))]
            al_t, ph_t, tn_t = target[0], target[1], target[2]
            al_l, ph_l, tn_l = target_len[0], target_len[1], target_len[2]

            # Step 2. Run our forward pass
            emissions, emission_len = model(waveform, wave_len)
            emissions = [torch.log_softmax(emission, dim=-1).permute(1,0,2) for emission in emissions]
            al, ph, tn = emissions
            al_loss = ctc_loss(al, al_t, emission_len, al_l)
            ph_loss = ctc_loss(ph, ph_t, emission_len, ph_l)
            tn_loss = ctc_loss(tn, tn_t, emission_len, tn_l)

            # Step 3. Run our backward pass
            optimizer.zero_grad()
            loss = al_loss + ph_loss + tn_loss
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
                save_temp(epoch, test_loss) # save temp checkpoint
                test_decoder(epoch, 5)
            
        # scheduler.step()
        save_checkpoint(epoch, mean(test_loss_q))
        save_log(f'e{epoch}.txt', ['============= Final Test ============='])
        test_decoder(epoch, 10) # run some sample prediction and see the result

train(NUM_EPOCHS)


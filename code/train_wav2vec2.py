import os
# from statistics import mean
import torch
import torchaudio
from pypinyin import lazy_pinyin, Style
from os.path import exists
from string import ascii_uppercase, ascii_lowercase

from utils.textDecoder import GreedyCTCDecoder, NaiveDecoder
from utils.helper import get_alphabet_labels, get_phoneme_labels, get_tone_labels, get_pitch_labels
from utils.dataset import *
from model.wav2vec2 import Wav2Vec2Builder

# 设置训练的参数
NUM_EPOCHS = 20
LOAD_PATH = './checkpoint/wav2vec/mul_all.pt' # checkpoint used if exist
LOG_PATH = './log/n1-' # log file
DATALOADER_WORKERS = 2 # dataloader workers
LOAD_OPTIMIZER = False # for momentun, Adam, ...
LOAD_INITIAL_EPOCH = False

# 使用HPC时，训练过程写入文件监控
def save_log(file_name, log, mode='a', path = LOG_PATH):
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

# 打印基础信息
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_log(f'e.txt', ['torch:', torch.__version__])
save_log(f'e.txt', ['torchaudio:', torchaudio.__version__])
save_log(f'e.txt', ['device:', device])
save_log(f'e.txt', ['HPC Node:', os.uname()[1]])
mean = lambda x: sum(x)/len(x)

# 训练时用的表，以及基于这些表的解码器
alphabet_labels = get_alphabet_labels()
phoneme_labels = get_phoneme_labels()
tone_labels = get_tone_labels()
alphabet_look_up = {s: i for i, s in enumerate(alphabet_labels)} # label转数字
phoneme_look_up = {s: i for i, s in enumerate(phoneme_labels)}
tone_look_up = {s: i for i, s in enumerate(tone_labels)}
alphabet_decoder = GreedyCTCDecoder(labels=alphabet_labels)
phoneme_decoder = GreedyCTCDecoder(labels=phoneme_labels)
tone_decoder = GreedyCTCDecoder(labels=tone_labels)

# 中文转换为这些label
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
    pinyin = lazy_pinyin(chinese, strict=True, style=Style.TONE3, neutral_tone_with_five=True, tone_sandhi=True, errors=lambda x: u'-')
    tone = [i[-1] for i in pinyin]
    return [i for i in  '|'.join(tone)]

# 综合以上label，转换器，decoder
labels_list = [alphabet_labels, phoneme_labels, tone_labels]
translators_list = [chinese2alphabet, chinese2phoneme, chinese2tone]
decoders = [alphabet_decoder, phoneme_decoder, tone_decoder]

# dataloader
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
        self.translators_list = translators_list
        self.alphabet_set = ascii_uppercase + ascii_lowercase

    def label2id(self, label_set:int, str):
        return [self.look_up_list[label_set][i] for i in str]

    def id2label(self, label_set:int, idcs):
        return ''.join([self.labels_list[label_set][i] for i in idcs])

    def contains_english(self, chinese):
        for i in range(len(chinese)):
            if chinese[i] in self.alphabet_set:
                return True
        return False

    def batch_filter(self, batch:list):
        # remove all audio with tag if audio length > threshold
        for i in range(len(batch)-1, -1, -1):
            if batch[i]['audio'].shape[-1] > self.threshold\
                or self.contains_english(batch[i]['chinese']): # remove all english 
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
            target_list = [self.label2id(label_set, self.translators_list[label_set](item['chinese'])) for item in batch]
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
        # self.threshold = audioDataset.dataset.threshold
        def max_threshold_in_concat_set(concat_set):
            max_threshold = 0
            for dataset in concat_set.datasets:
                max_threshold = max(dataset.dataset.threshold, max_threshold)
            return max_threshold
        self.threshold = max_threshold_in_concat_set(audioDataset)
        return DataLoader(audioDataset, batch_size,
                            shuffle, num_workers=self.num_workers, collate_fn=self.collate_wrapper)

def raw_audio_transform(sample, sample_rate=None):
        audio = sample['audio']
        audio = audio / torch.abs(audio).max()*0.15
        sample['audio'] = audio
        sample['chinese'] = sample['text']
        return sample

def ai_shell_3_transform(sample, sample_rate=None):
        audio = sample['audio']
        audio = audio / torch.abs(audio).max()*0.15
        sample['audio'] = audio
        text = sample['text']
        text = text.split(' ')
        chinese = [text[i] for i in range(len(text)) if i%2==0]
        sample['chinese'] = ''.join(chinese)
        return sample

dataset1 = PrimeWordsDataset('/scratch/bh2283/data/primewords_md_2018_set1/', transform=raw_audio_transform)
dataset2 = STCMDSDataset('/ST-CMDS-20170001_1-OS/', transform=raw_audio_transform) # singularity usage only
dataset3 = AiShellDataset('/scratch/bh2283/data/data_aishell/', transform=raw_audio_transform)
dataset4 = AiDataTangDataset('/aidatatang_200zh/', transform=raw_audio_transform) # singularity usage only
dataset5 = AiShell3Dataset('/data_aishell3/train/', transform=ai_shell_3_transform) # singularity usage only
dataset6 = SpeechOceanDataset('/scratch/bh2283/data/zhspeechocean/', transform=raw_audio_transform)
datasets = [dataset1, dataset2, dataset3, dataset4, dataset5, dataset6]

train_sets = []
test_sets = []
for dataset in datasets:
    train_set, test_set = dataset.split()
    train_sets.append(train_set)
    test_sets.append(test_set)

train_set = torch.utils.data.ConcatDataset(train_sets)
test_set = torch.utils.data.ConcatDataset(test_sets)

labels_sizes = [len(labels) for labels in labels_list]
builder = Wav2Vec2Builder(torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_EN, labels_sizes)
k_size = builder.kernel_size

# batch_size = int(train_set.dataset.batch_size*0.8) # tain batch size
batch_size = 24
test_batch = batch_size # test batch size, keep bs small to save memory
loaderGenerator = MultiTaskRawLoaderGenerator(labels_list, translators_list, k_size, num_workers=DATALOADER_WORKERS)
train_loader = loaderGenerator.dataloader(train_set, batch_size)
test_loader = loaderGenerator.dataloader(test_set, test_batch, shuffle=False)
save_log(f'e.txt', ['train_set:', len(train_set), 'test_set:',len(test_set)])
save_log(f'e.txt', ['train batch_size:', batch_size, ', test batch_size', test_batch])

# 模型初始化
save_log(f'e.txt', ['Init Model ...'])
from model.wav2vec2 import Wav2Vec2Builder
model = builder.get_model()
save_log(f'e.txt', ['k_size:', builder.kernel_size])

# 不修改feature_extractor，因为在最底层，可以不回传梯度。只对encoder和aux做梯度下降
for param in model.feature_extractor.parameters():
    param.requires_grad = False
model = model.to(device)
params = list(model.encoder.parameters()) + list(model.aux.parameters())
# optimizer = torch.optim.Adam(params, lr=0.00001)
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.8)
ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
initial_epoch = 0

# 加载记录点
def load_checkpoint(path):
    if exists(path):
        print('file',path,'exist, load checkpoint...')
        checkpoint = torch.load(path, map_location=device)
        if 'model_aux_dict' in checkpoint:
            model.aux.load_state_dict(checkpoint['model_aux_dict'])
        if 'model_encoder_dict' in checkpoint:
            model.encoder.load_state_dict(checkpoint['model_encoder_dict'])
        if 'optimizer_state_dict' in checkpoint and LOAD_OPTIMIZER:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch'] if LOAD_INITIAL_EPOCH else 0
        loss = checkpoint['loss']
        print(f'initial_epoch: {initial_epoch}, loss: {loss}')


# 测试集上打印一些示例，可以用肉眼判断训练情况好坏
def test_decoder(epoch, k):
    model.eval()
    with torch.no_grad():
        get_transcript = lambda x, emissions: decoders[x](torch.log_softmax(emissions[x][0], dim=-1).cpu().detach())
        get_naive_char = lambda x, emissions: NaiveDecoder(labels=alphabet_labels)(torch.log_softmax(emissions[x][0], dim=-1).cpu().detach())
        for i in range(k):
            sample = test_set[i]
            print(i, sample['audio'].shape)
            save_log(f'e{epoch}.txt', ['Chinese:', sample['chinese']])
            waveform = sample['audio']
            emissions, _ = model(waveform.to(device))
            save_log(f'e{epoch}.txt', [
                'Char level:', ''.join(get_transcript(0, emissions)),
                '\nPhon level:', ''.join(get_transcript(1, emissions)),
                '\nTone level:', ''.join(get_transcript(2, emissions)),
                '\nC-Nv level:', get_naive_char(0, emissions),
                ])


# 训练过程中勤快地保存数据，并且每个epoch保存一个单独的数据（不覆盖）
def dump_model(EPOCH, LOSS, PATH):
    torch.save({
            'epoch': EPOCH,
            'model_aux_dict': model.aux.state_dict(),
            'model_encoder_dict': model.encoder.state_dict(),
            'model_feature_extractor_dict': model.feature_extractor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

def save_temp(EPOCH, LOSS):
    PATH = f"./checkpoint/wav2vec/model_temp.pt"
    dump_model(EPOCH, LOSS, PATH)
    
def save_checkpoint(EPOCH, LOSS):
    PATH = f"./checkpoint/wav2vec/model_{EPOCH}_{'%.3f' % LOSS}.pt"
    dump_model(EPOCH, LOSS, PATH)

# 测试集上的多个loss计算，于训练集对比
def test():
    model.eval()
    losses = []
    al_losses = []
    ph_losses = []
    tn_losses = []
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
            al_losses.append(al_loss.item())
            ph_losses.append(ph_loss.item())
            tn_losses.append(tn_loss.item())
    return mean(losses), mean(al_losses), mean(ph_losses), mean(tn_losses)

def blank_loss(emission, emission_len, blank_id=0, margin=0):
    # input emission with shape [time, bs, class]
    # input emission_len with shape [bs]
    blank_emission_p = torch.exp(emission[:, :, blank_id]) # probability of blank
    blank_emission_p = torch.relu(blank_emission_p - margin) # dismiss if less than margin
    mask = torch.arange(max(emission_len))[:, None].to(device) < emission_len[None, :]
    # print(blank_emission_p)
    # print(mask)
    blank_emission_p_masked = blank_emission_p * mask # apply mask
    # print(blank_emission_p_masked)
    mean_axis = torch.sum(blank_emission_p_masked, dim=0)/emission_len # mean blank p in axis
    # print(mean_axis)
    return torch.mean(mean_axis)

# Training the model
def train(epoch=1):
    train_loss_q = []
    test_loss_q = []

    al_margin = 1.0/len(alphabet_labels)
    ph_margin = 1.0/len(phoneme_labels)
    tn_margin = 1.0/len(tone_labels)
    
    for epoch in range(initial_epoch, epoch):
        
        batch_train_loss, b_al, b_ph, b_tn, b_blank = [], [], [], [], []
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

            b_loss = blank_loss(al, emission_len, margin=al_margin) + blank_loss(ph, emission_len, margin=ph_margin) + blank_loss(tn, emission_len, margin=tn_margin)
            splite_loss = blank_loss(al, emission_len, 1, al_margin) + blank_loss(ph, emission_len, 1, ph_margin) + blank_loss(tn, emission_len, 1, tn_margin)
            b_loss = b_loss * 0.5
            splite_loss = splite_loss * 0.5
            
            # Step 3. Run our backward pass
            optimizer.zero_grad()
            loss = al_loss + ph_loss + tn_loss + b_loss + splite_loss
            loss.backward()
            optimizer.step()

            # print(al_loss.item(), ph_loss.item(), tn_loss.item(), b_loss.item())
            # exit()


            if loss.item()!=loss.item(): # if loss == NaN, break
                print('NaN hit!')
                exit()
            
            batch_train_loss.append(loss.item())
            b_al.append(al_loss.item())
            b_ph.append(ph_loss.item())
            b_tn.append(tn_loss.item())
            b_blank.append(b_loss.item())

            if i_batch % (5000 // batch_size) == 0: # log about each 1000 data
                test_loss, t_al, t_ph, t_tn = test()
                # test_loss = 0
                batch_train_loss = mean(batch_train_loss)
                b_al, b_ph, b_tn, b_blank = mean(b_al), mean(b_ph), mean(b_tn), mean(b_blank)

                test_loss_q.append(test_loss)
                train_loss_q.append(batch_train_loss)
                save_log(f'e{epoch}.txt', ['🟣 epoch', epoch, 'data', i_batch*batch_size, 
                    'lr', scheduler.get_last_lr(), 
                    '[train_loss:{:.3f}]'.format(batch_train_loss), 
                    'al:{:.3f}, ph:{:.3f}, tn:{:.3f}, blank:{:.3f}'.format(b_al, b_ph, b_tn, b_blank),
                    '[test_los:{:.3f}]'.format(test_loss),
                    'al:{:.3f}, ph:{:.3f}, tn:{:.3f}'.format(t_al, t_ph, t_tn),
                    ])
                save_temp(epoch, test_loss) # save temp checkpoint
                batch_train_loss, b_al, b_ph, b_tn, b_blank = [], [], [], [], []
            if i_batch % (40000 // batch_size) == 0:
                test_decoder(epoch, 2)
            
        scheduler.step()
        save_checkpoint(epoch, mean(test_loss_q))
        save_log(f'e{epoch}.txt', ['============= Final Test ============='])
        test_decoder(epoch, 10) # run some sample prediction and see the result

if __name__ == '__main__':
    save_log(f'e.txt', ['Loading Checkpoint ...'])
    load_checkpoint(LOAD_PATH)
    test_decoder('', 2)
    save_log(f'e.txt', ['initial test loss:', test()])
    save_log(f'e.txt', ['Start training ...'])
    train(NUM_EPOCHS)


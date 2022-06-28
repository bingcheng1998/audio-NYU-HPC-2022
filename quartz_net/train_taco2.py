from dataset import OpencpopDataset, MusicLoaderGenerator
from helper import parser_line, merge_note, get_pitch_labels, get_transposed_phoneme_labels, print_all
import torchaudio
from torchaudio.models.tacotron2 import Tacotron2, _get_mask_from_lengths, _Decoder, _Encoder, _Postnet
from torchaudio.pipelines._tts.utils import _get_taco_params
import torch
from torch import Tensor
from typing import Tuple, List, Optional, Union, overload
import os
from os.path import exists

BATCH_SIZE = 16
LOG_DIR = './log/tacotron-4-'
DATALOADER_WORKERS = 8
LEARNING_RATE = 0.00001
# LOAD_PATH = './checkpoint/pre.pt'
LOAD_PATH = './checkpoint/model_temp.pt'
SAMPLE_RATE= 22050

def save_log(file_name, log, mode='a', path = LOG_DIR):
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

# torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_log(f'e.txt', ['torch:', torch.__version__])
save_log(f'e.txt', ['torchaudio:', torchaudio.__version__])
save_log(f'e.txt', ['device:', device])
save_log(f'e.txt', ['HPC Node:', os.uname()[1]])

def dataset_transform(sample, sample_rate=None):
    id, text, phoneme, note, note_duration, phoneme_duration, slur_note = parser_line(sample['text'])
    text_with_p, phoneme, note, note_duration, slur_note = merge_note(text, phoneme, note, note_duration, slur_note)
    sample['chinese'] = text_with_p
    sample['phoneme'] = phoneme
    sample['note'] = note
    sample['duration'] = note_duration
    sample['slur'] = slur_note
    return sample

dataset = OpencpopDataset('/scratch/bh2283/data/opencpop/segments/', transform=dataset_transform, sample_rate=SAMPLE_RATE)
train_set, test_set = dataset.split()
save_log(f'e.txt', ['train_set:', len(train_set), 'test_set:',len(test_set)])

note_labels = get_pitch_labels()
phoneme_labels = get_transposed_phoneme_labels()
slur_labels = [0, 1]

labels = (
    phoneme_labels,
    note_labels,
    slur_labels
)
loaderGenerator = MusicLoaderGenerator(labels, DATALOADER_WORKERS)
batch_size = BATCH_SIZE if device == torch.device("cuda") else 4
train_loader = loaderGenerator.dataloader(train_set, batch_size=batch_size)
test_loader = loaderGenerator.dataloader(test_set, batch_size=batch_size)

class TacotronTail(Tacotron2):
    def __init__(
        self,
        labels_lens: dict,
        decoder = None,
        postnet = None,
    ) -> None:
        _tacotron2_params=_get_taco_params(n_symbols=5) # ignore n_symbols, encoder not used 
        _tacotron2_params["mask_padding"] = True
        super().__init__(**_tacotron2_params)

        embedding_dim = _tacotron2_params['encoder_embedding_dim']
        self.embeddings = {
            key: torch.nn.Embedding(value, embedding_dim)
            for key, value in labels_lens.items()
        }
        self.embedding_register = torch.nn.ModuleList(self.embeddings.values())
        # 将embedding注册进模型，不确定是否复制，需要在实践中测试
        if decoder is not None:
            self.decoder: _Decoder = decoder
        if postnet is not None:
            self.postnet: _Postnet = postnet
        self.reduce_phoneme = lambda x: torch.sum(x, 1) if len(x.shape)==3 else x
        self.decoder.decoder_max_step = int(4 * 22050 / 256)
        self.version = '0.01'

    def forward(
        self,
        inputs: dict,
    ):
        embedded_inputs = [
            self.reduce_phoneme(self.embeddings[key](inputs[key])) for key in self.embeddings.keys()
        ]
        embedded_inputs = torch.stack(embedded_inputs).sum(0).unsqueeze(1) # [bs, 1, emb_size]
        # print('embedded_inputs', embedded_inputs.shape)
        mel_specgram = inputs['mel'] # (n_batch, ``n_mels``, max of ``mel_specgram_lengths``)
        # print('mel_specgram', mel_specgram.shape)
        mel_specgram_lengths = inputs['mel_len']
        mel_specgram, gate_outputs, alignments = self.decoder(
            embedded_inputs, mel_specgram, memory_lengths=torch.ones_like(mel_specgram_lengths),
        )

        mel_specgram_postnet = self.postnet(mel_specgram)
        mel_specgram_postnet = mel_specgram + mel_specgram_postnet

        if self.mask_padding:
            mask = _get_mask_from_lengths(mel_specgram_lengths)
            mask = mask.expand(self.n_mels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            mel_specgram = mel_specgram.masked_fill(mask, 0.0)
            mel_specgram_postnet = mel_specgram_postnet.masked_fill(mask, 0.0)
            # gate_outputs.masked_fill_(mask[:, 0, :], 1e3)

        return mel_specgram, mel_specgram_postnet, gate_outputs, alignments

    @torch.jit.export
    def infer(
        self, 
        inputs: dict,
        ) -> Tuple[Tensor, Tensor, Tensor]:

        embedded_inputs = [
            self.reduce_phoneme(self.embeddings[key](inputs[key])) for key in self.embeddings.keys()
        ]
        embedded_inputs = torch.stack(embedded_inputs).sum(0).unsqueeze(1) # [bs, 1, emb_size]
        # print('embedded_inputs', embedded_inputs.shape)
        
        n_batch = embedded_inputs.shape[0]
        mel_specgram, mel_specgram_lengths, _, alignments = \
            self.decoder.infer(embedded_inputs, memory_lengths=torch.ones(n_batch))

        mel_outputs_postnet = self.postnet(mel_specgram)
        mel_outputs_postnet = mel_specgram + mel_outputs_postnet

        alignments = alignments.unfold(1, n_batch, n_batch).transpose(0, 2)

        return mel_outputs_postnet, mel_specgram_lengths, alignments

labels_lens = {
    'audio_duration_quant': 130, # 这个是量化后的计算结果
    'phoneme': len(phoneme_labels), # 拼音
    'phoneme_pre': len(phoneme_labels), # 前一个汉字的拼音
    'phoneme_post': len(phoneme_labels), # 后一个汉字的拼音
    'note': len(note_labels), # 音调音符
    'note_pre': len(note_labels),
    'note_post': len(note_labels),
    'slur': len(slur_labels), # 是否为延长音
}
model = TacotronTail(labels_lens).to(device)

def load_checkpoint(path):
    if exists(path):
        save_log(f'e.txt', ['path', path, 'exist, loading...'])
        checkpoint = torch.load(path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            # model.decoder.load_state_dict(checkpoint['model_state_dict'], strict=False) # , strict=False?
            # model.postnet.load_state_dict(checkpoint['model_state_dict'], strict=False)

load_checkpoint(LOAD_PATH)

params = model.parameters()
# params = list(model.embedding.parameters())+list(model.encoder.parameters())+list(model.speaker_encoder.parameters())
# optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.5)
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
initial_epoch = 0
mse_loss = torch.nn.MSELoss()
bce_loss = torch.nn.BCELoss()
cos_loss = torch.nn.CosineEmbeddingLoss()
mean = lambda x: sum(x)/len(x)

def dump_model(EPOCH, LOSS, PATH):
    if device == torch.device("cuda"):
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

def save_temp(EPOCH, LOSS):
    PATH = f"./checkpoint/model_temp.pt"
    dump_model(EPOCH, LOSS, PATH)
    
def save_checkpoint(EPOCH, LOSS):
    PATH = f"./checkpoint/model_{EPOCH}_{'%.3f' % LOSS}.pt"
    dump_model(EPOCH, LOSS, PATH)

def threshold_mse_loss(mel1, mel2, threshold=0.2, mask_fill_val=0):
    # threshold should be in [0,1]
    # the larger the threshold, the smaller area selected
    # mel1 [mel_bins, time], mel1 and mel2 should have same time
    assert mel1.shape[1] == mel2.shape[1]
    mask1 = mel1 < mel1.mean() + mel1.var()*threshold
    mask2 = mel2 < mel2.mean() + mel2.var()*threshold
    and_mask = mask1.bitwise_and(mask2)
    return mel1.masked_fill(and_mask, mask_fill_val), mel2.masked_fill(and_mask, mask_fill_val)

def mse_total_loss(mels_tensor, org_mel, pos_mel):
    loss1 = mse_loss(mels_tensor, org_mel)
    loss1 += mse_loss(mels_tensor, pos_mel)
    org_mel_masked, pos_mel_masked = threshold_mse_loss(mels_tensor, pos_mel, 0.2)
    loss1 += mse_loss(org_mel_masked, pos_mel_masked)*2
    org_mel_masked, pos_mel_masked = threshold_mse_loss(mels_tensor, pos_mel, 0.8)
    loss1 += mse_loss(org_mel_masked, pos_mel_masked)*5
    org_mel_masked, pos_mel_masked = threshold_mse_loss(mels_tensor, pos_mel, 2)
    loss1 += mse_loss(org_mel_masked, pos_mel_masked)*10
    return loss1

def test():
    model.eval()
    batch_test_loss = []
    batch_bce_loss = []
    for i_batch, sample_batched in enumerate(test_loader):
        sample_batched = {
            k:v.to(device) for k, v in sample_batched.items() if isinstance(v, torch.Tensor)
        }
        # Step 1. Prepare Data
        mels_tensor = sample_batched['mel'].to(device) # [bs, mel_bins, L]
        mel_length = sample_batched['mel_len'].to(device)
        # Step 2. Run our forward pass
        org_mel, pos_mel, stop_token, _ = model.forward(sample_batched)
        loss1 = mse_total_loss(mels_tensor, org_mel, pos_mel)
        true_stop_token = torch.zeros(stop_token.shape).to(device)
        for i in range(true_stop_token.shape[0]):
            true_stop_token[i][mel_length[i]:]+=1.0
        loss2 = bce_loss(torch.sigmoid(stop_token), true_stop_token)
        # Step 3. add all loss
        loss = loss1 + loss2
        batch_bce_loss.append(loss2.item())
        batch_test_loss.append(loss.item())
    return mean(batch_test_loss), mean(batch_bce_loss)

def train(epoch=1):
    train_loss_q = []
    test_loss_q = []
    for epoch in range(initial_epoch, epoch):
        batch_train_loss = []
        for i_batch, sample_batched in enumerate(train_loader):
            model.train()
            sample_batched = {
                k:v.to(device) for k, v in sample_batched.items() if isinstance(v, torch.Tensor)
            }
            # Step 1. Prepare Data
            mels_tensor = sample_batched['mel'].to(device) # [bs, mel_bins, L]
            mel_length = sample_batched['mel_len'].to(device)

            # Step 2. Run our forward pass
            # mask = _get_mask_from_lengths(mel_length)
            # mask = mask.expand(80, mask.size(0), mask.size(1))
            # mask = mask.permute(1, 0, 2)
            # mels_tensor = mels_tensor.masked_fill(mask, 0.0)

            org_mel, pos_mel, stop_token, _ = model.forward(sample_batched)
            loss1 = mse_total_loss(mels_tensor, org_mel, pos_mel)

            true_stop_token = torch.zeros(stop_token.shape).to(device)
            for i in range(true_stop_token.shape[0]):
                true_stop_token[i][mel_length[i]:]+=1.0
            loss2 = bce_loss(torch.sigmoid(stop_token), true_stop_token)
            
            # Step 3. Run our backward pass
            optimizer.zero_grad()
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            if loss.item()!=loss.item(): # if loss == NaN, break
                print('NaN hit!')
                exit()
            
            batch_train_loss.append(loss.item())

            if i_batch % (300 // BATCH_SIZE) == 0: # log about each n data
                test_loss, test_bce_loss = test()
                # test_loss = 0
                train_loss = mean(batch_train_loss)
                test_loss_q.append(test_loss)
                train_loss_q.append(train_loss)
                save_log(f'e{epoch}.txt', ['🟣 epoch', epoch, 'data', i_batch*BATCH_SIZE, 
                    'lr', scheduler.get_last_lr(), 
                    'train_loss', '{:.3f}'.format(train_loss), 
                    'test_loss', '{:.3f}'.format(test_loss), 
                    'test_bce_loss', '{:.3f}'.format(test_bce_loss)])
                save_temp(epoch, test_loss) # save temp checkpoint
            exit()
            
        # scheduler.step()
        save_checkpoint(epoch, mean(test_loss_q))
        save_log(f'e{epoch}.txt', ['============= Final Test ============='])


train(200)
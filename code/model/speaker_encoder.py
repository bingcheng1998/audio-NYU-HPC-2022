import torch
import torch.nn.functional as F
import torch.nn as nn

class SpeakerEncoder(nn.Module):
    def __init__(self, n_mel, hidden_dim, out_dim) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_mel,
                            hidden_size=hidden_dim, 
                            num_layers=3, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=hidden_dim, 
                                out_features=out_dim)
        self.relu = torch.nn.ReLU()

        self.similarity_weight = nn.Parameter(torch.tensor([10.])) # ignore
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])) # ignore

    def forward(self, utterances, lens, hidden_init=None):
        # utterances [bs, L, mel_in]
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        # print('hidden', hidden.shape)
        # print('out', out.shape) # [bs, L, hidden_dim]
        # We take only the hidden state of the lens-1 layer
        index_out = [out[i][lens[i]-1].unsqueeze(0) for i in range(out.shape[0])]
        embeds_raw = torch.concat(index_out, dim=0)
        # print('embeds_raw', embeds_raw.shape)
        # L2-normalize it 
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        
        return embeds

if __name__ == '__main__':
    model = SpeakerEncoder(4, 8, 6)
    spec = torch.randn(5, 11, 4) # bs, L, mel_in
    lens=torch.tensor([1,2,3,4,5])
    out = model(spec, lens) # bs, out_dim
    print(out.shape)

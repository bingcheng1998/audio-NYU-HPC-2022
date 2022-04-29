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

    def forward(self, utterances, hidden_init=None):
        # spec [bs x n_mel x l]
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        
        return embeds

if __name__ == '__main__':
    model = SpeakerEncoder(40, 256, 256)
    spec = torch.randn(5, 99, 40) # bs, L, in
    out = model(spec) # bs, L, out
    print(out.shape)

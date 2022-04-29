import torch
import torch.nn.functional as F
import torch.nn as nn

class speakerEncoder(nn.Module):
    def __init__(self, n_mel, hidden_dim, out_dim) -> None:
        super().__init__()
        self.rnn1 = nn.LSTM(n_mel, hidden_dim, batch_first=True)
        self.rnn2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, out_dim)

    def forward(self, spec):
        # spec [bs x n_mel x l]
        rnn1_out, _ = self.rnn1(spec)
        rnn2_out, _ = self.rnn2(rnn1_out)
        out = self.hidden2out(rnn2_out)
        return out

if __name__ == '__main__':
    speaker_encoder = speakerEncoder(80, 256, 128)
    spec = torch.randn(1, 99, 80) # bs, L, in
    out = speaker_encoder(spec) # bs, L, out
    print(out.shape)

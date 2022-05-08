import torch

def trim_mel_silence(mel_in, min_range=10, ratio=5.0, min_threshold=0.1):
    # mel_in: [mel_bin, L]
    # you should make sure the top min_range mel columns are silent
    assert ratio >= 1.0, f'get ratio {ratio}, but it should be larger than 1.0, and 5.0 is recommended.'
    mel_sum = torch.sum(mel_in, dim=0) # [L]
    top10 = torch.sum(mel_sum[:min_range])
    threshold = min(top10*ratio/min_range, min_threshold)
    # print('threshold', threshold)
    mel_cut = (mel_sum < threshold) * 1
    for i in range(len(mel_cut)):
        if (mel_cut[i]==1 and i > 0):
            mel_cut[i]=mel_cut[i]+mel_cut[i-1]
    i = len(mel_sum)-1
    while i > min_range:
        if (mel_cut[i] > min_range):
            mel_in = torch.concat([mel_in[:, :i-mel_cut[i]+1], mel_in[:, i+1:]], dim=1)
            i -= mel_cut[i]
        else:
            i-=1
    return mel_in
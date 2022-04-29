from torch import nn
import torch
from model.layers import MainBlock
from model.utils import init_weights
from model.config import quartznet5x5_config

class QuartzNet(nn.Module):
    def __init__(
            self,
            n_mels,
            num_classes,
            model_config=quartznet5x5_config,
            activation='relu',
            normalization_mode="batch",
            norm_groups=-1,
            frame_splicing=1,
            init_mode='xavier_uniform',
            **kwargs
    ):
        super(QuartzNet, self).__init__()
        feat_in = n_mels
        vocab_size = num_classes
        feat_in = feat_in * frame_splicing
        self.stride = 1

        residual_panes = []
        layers = []
        for lcfg in model_config:
            self.stride *= lcfg['stride']

            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            residual = lcfg.get('residual', True)
            layers.append(
                MainBlock(feat_in,
                    lcfg['filters'],
                    repeat=lcfg['repeat'],
                    kernel_size=lcfg['kernel'],
                    stride=lcfg['stride'],
                    dilation=lcfg['dilation'],
                    dropout=lcfg['dropout'] if 'dropout' in lcfg else 0.0,
                    residual=residual,
                    groups=groups,
                    separable=separable,
                    normalization=normalization_mode,
                    norm_groups=norm_groups,
                    activation=activation))
            feat_in = lcfg['filters']

        self.encoder = nn.Sequential(*layers)
        self.classify = nn.Conv1d(1024, vocab_size,
                      kernel_size=1, bias=True)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, x, lengths = None):
        feat = self.encoder(x)
        # BxCxT
        if lengths is not None:
            stride = self.encoder[0].net[0][0].stride[0]
            lengths = torch.ceil(lengths/stride).int()
        return self.classify(feat), lengths

    def load_weights(self, path, map_location='cpu'):
        weights = torch.load(path, map_location=map_location)
        print(self.load_state_dict(weights, strict=False))
from dataclasses import dataclass
from typing import Dict, Tuple, Any
import torch
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torchaudio._internal import load_state_dict_from_url
# from torchaudio.models import wav2vec2_model, Wav2Vec2Model

class Wav2Vec2Builder:
    def __init__(self, bundle, out_sizes: list):
        super().__init__()
        base_model = bundle.get_model()
        self.feature_extractor = base_model.feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False # do not change feature_extractor
        self.encoder = base_model.encoder
        self.in_features = base_model.encoder.transformer.layers[-1].final_layer_norm.normalized_shape[0]
        self._set_aux(out_sizes)
        self.kernel_size = base_model.feature_extractor.conv_layers[0].conv.kernel_size[0]
    
    def _set_aux(self, out_sizes: list):
        self.aux = []
        for out_size in out_sizes:
            self.aux.append(
                torch.nn.Linear(self.in_features, out_size, bias=True)
            )
        self.aux = torch.nn.ModuleList(self.aux)

    def _get_url_state_dict(self, url):
        state_dict = load_state_dict_from_url(url)
        return state_dict

    def _get_local_state_dict(self, path):
        pass

    def get_model(self, checkpoint=None, url=None):
        model = Wave2vec2(self.feature_extractor, self.encoder, self.aux)
        if checkpoint is not None:
            model.load_state_dict(self._get_local_state_dict(checkpoint))
        elif url is not None:
            model.load_state_dict(self._get_url_state_dict(url))
        model.eval()
        return model

class Wave2vec2(torch.nn.Module):
    def __init__(self, feature_extractor, encoder, aux):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.aux = aux
        self.f_mask = FrequencyMasking(freq_mask_param=15)
        self.t_mask = TimeMasking(time_mask_param=15)

    def forward(self, x, lengths=None):
        x, lengths = self.feature_extractor(x, lengths)
        x = self.t_mask(x) # time mask
        x = self.t_mask(x) # 2nd time mask
        x = self.f_mask(x) # feature mask
        x = self.encoder(x, lengths)
        output = []
        for aux_i in self.aux:
            output.append(aux_i(x))
        return output, lengths
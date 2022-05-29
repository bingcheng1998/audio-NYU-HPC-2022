from torchaudio.pipelines._tts.utils import _get_taco_params
from model.official_tacotron2_adapted import Tacotron2, _get_mask_from_lengths, _Decoder, _Encoder, _Postnet
import torch
from torch import Tensor
from typing import Tuple, List, Optional, Union, overload
from model.speaker_encoder import SpeakerEncoder

class My_Decoder(_Decoder):
    @overload
    def forward(
        self, memory: Tensor, mel_specgram_truth: Tensor, memory_lengths: Tensor, alpha: int = 0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Decoder forward pass for training.

        Args:
            memory (Tensor): Encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).
            mel_specgram_truth (Tensor): Decoder ground-truth mel-specs for teacher forcing
                with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).
            memory_lengths (Tensor): Encoder output lengths for attention masking
                (the same as ``text_lengths``) with shape (n_batch, ).

        Returns:
            mel_specgram (Tensor): Predicted mel spectrogram
                with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).
            gate_outputs (Tensor): Predicted stop token for each timestep
                with shape (n_batch,  max of ``mel_specgram_lengths``).
            alignments (Tensor): Sequence of attention weights from the decoder
                with shape (n_batch,  max of ``mel_specgram_lengths``, max of ``text_lengths``).
        """

        decoder_input = self._get_initial_frame(memory).unsqueeze(0) # 添加全0向量补齐长度
        decoder_inputs = self._parse_decoder_inputs(mel_specgram_truth)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0) # 补齐后有了全0初始帧
        decoder_inputs = self.prenet(decoder_inputs)

        mask = _get_mask_from_lengths(memory_lengths)

        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self._initialize_decoder_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        mel_output = self._get_go_frame(memory)
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            decoder_input2 = self.prenet(mel_output)
            decoder_input = alpha * decoder_input2 + (1-alpha) * decoder_input
            (
                mel_output,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask,
            )

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_specgram, gate_outputs, alignments = self._parse_decoder_outputs(
            torch.stack(mel_outputs), torch.stack(gate_outputs), torch.stack(alignments)
        )

        return mel_specgram, gate_outputs, alignments

class MyTacotron2(Tacotron2):
    def __init__(
        self,
        labels,
        speaker_emb_size = 128,
        decoder = None,
        postnet = None,
    ) -> None:
        _tacotron2_params=_get_taco_params(n_symbols=len(labels))
        super().__init__(**_tacotron2_params)

        cutted_encoder_embedding_dim = _tacotron2_params['encoder_embedding_dim'] - speaker_emb_size
        self.speaker_encoder = SpeakerEncoder(_tacotron2_params['n_mels'], 256, speaker_emb_size)
        self.embedding = torch.nn.Embedding(_tacotron2_params['n_symbol'], cutted_encoder_embedding_dim)
        self.encoder: _Encoder = _Encoder(cutted_encoder_embedding_dim, _tacotron2_params['encoder_n_convolution'], _tacotron2_params['encoder_kernel_size'])
        if decoder is not None:
            self.decoder: _Decoder = decoder
        if postnet is not None:
            self.postnet: _Postnet = postnet
        self.speaker_emb_size = speaker_emb_size
        self.version = '0.05'
    
    def forward(
        self,
        tokens: Tensor,
        token_lengths: Tensor,
        mel_specgram: Tensor,
        mel_specgram_lengths: Tensor,
        speaker_emb: Optional[Tensor]=None,
        alpha: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Pass the input through the Tacotron2 model. This is in teacher
        forcing mode, which is generally used for training.

        The input ``tokens`` should be padded with zeros to length max of ``token_lengths``.
        The input ``mel_specgram`` should be padded with zeros to length max of ``mel_specgram_lengths``.

        Args:
            tokens (Tensor): The input tokens to Tacotron2 with shape `(n_batch, max of token_lengths)`.
            token_lengths (Tensor): The valid length of each sample in ``tokens`` with shape `(n_batch, )`.
            mel_specgram (Tensor): The target mel spectrogram
                with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
            mel_specgram_lengths (Tensor): The length of each mel spectrogram with shape `(n_batch, )`.

        Returns:
            [Tensor, Tensor, Tensor, Tensor]:
                Tensor
                    Mel spectrogram before Postnet with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
                Tensor
                    Mel spectrogram after Postnet with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
                Tensor
                    The output for stop token at each time step with shape `(n_batch, max of mel_specgram_lengths)`.
                Tensor
                    Sequence of attention weights from the decoder with
                    shape `(n_batch, max of mel_specgram_lengths, max of token_lengths)`.
        """

        embedded_inputs = self.embedding(tokens).transpose(1, 2) # (bs, encoder_embedding_dim, L)

        encoder_outputs = self.encoder(embedded_inputs, token_lengths) # (bs, L, encoder_embedding_dim)

        # My change: calculate speaker_emb, and put it inside encoder_outputs (concat)
        if speaker_emb is None:
            speaker_emb = self.speaker_encoder(mel_specgram.transpose(1,2), mel_specgram_lengths) # input: [bs, L, mel_in]; output: [bs, speaker_emb_size]
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, encoder_outputs.shape[1], -1) # (bs, L, speaker_emb_size)
        # print('xx', encoder_outputs.shape, speaker_emb.shape)
        encoder_outputs = torch.concat([encoder_outputs, speaker_emb], -1) # (bs, L, speaker_emb_size+encoder_embedding_dim)
        
        # end this part

        mel_specgram, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_specgram, memory_lengths=token_lengths, alpha=alpha
        )

        mel_specgram_postnet = self.postnet(mel_specgram)
        mel_specgram_postnet = mel_specgram + mel_specgram_postnet

        if self.mask_padding:
            mask = _get_mask_from_lengths(mel_specgram_lengths)
            mask = mask.expand(self.n_mels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            mel_specgram.masked_fill_(mask, 0.0)
            mel_specgram_postnet.masked_fill_(mask, 0.0)
            gate_outputs.masked_fill_(mask[:, 0, :], 1e3)

        return mel_specgram, mel_specgram_postnet, gate_outputs, alignments

    @torch.jit.export
    def infer(self, tokens: Tensor, lengths: Optional[Tensor] = None, speaker_emb: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Using Tacotron2 for inference. The input is a batch of encoded
        sentences (``tokens``) and its corresponding lengths (``lengths``). The
        output is the generated mel spectrograms, its corresponding lengths, and
        the attention weights from the decoder.

        The input `tokens` should be padded with zeros to length max of ``lengths``.

        Args:
            tokens (Tensor): The input tokens to Tacotron2 with shape `(n_batch, max of lengths)`.
            lengths (Tensor or None, optional):
                The valid length of each sample in ``tokens`` with shape `(n_batch, )`.
                If ``None``, it is assumed that the all the tokens are valid. Default: ``None``

        Returns:
            (Tensor, Tensor, Tensor):
                Tensor
                    The predicted mel spectrogram with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
                Tensor
                    The length of the predicted mel spectrogram with shape `(n_batch, )`.
                Tensor
                    Sequence of attention weights from the decoder with shape
                    `(n_batch, max of mel_specgram_lengths, max of lengths)`.
        """
        n_batch, max_length = tokens.shape
        if lengths is None:
            lengths = torch.tensor([max_length]).expand(n_batch).to(tokens.device, tokens.dtype)

        assert lengths is not None  # For TorchScript compiler

        embedded_inputs = self.embedding(tokens).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, lengths)
        
        # My change: calculate speaker_emb, and put it inside encoder_outputs (concat)
        # speaker_emb = self.speaker_encoder(mel_specgram, mel_specgram_lengths) # input: [bs, L, mel_in]; output: [bs, out_dim]
        if speaker_emb is None:
            speaker_emb = torch.zeros([encoder_outputs.shape[0], self.speaker_emb_size])
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, encoder_outputs.shape[1], -1) # (bs, L, out_dim)
        encoder_outputs = torch.concat([encoder_outputs, speaker_emb], -1)
        # end this part

        mel_specgram, mel_specgram_lengths, _, alignments = self.decoder.infer(encoder_outputs, lengths)

        mel_outputs_postnet = self.postnet(mel_specgram)
        mel_outputs_postnet = mel_specgram + mel_outputs_postnet

        alignments = alignments.unfold(1, n_batch, n_batch).transpose(0, 2)

        return mel_outputs_postnet, mel_specgram_lengths, alignments
import whisper
from whisper.model import AudioEncoder, MultiHeadAttention, ResidualAttentionBlock, TextDecoder
from typing import Optional
import torch
from torch import Tensor
from torch import nn

class AudioEncoderTensorCache(nn.Module):
    def __init__(self, inAudioEncoder: AudioEncoder, inTextDecoder: TextDecoder):
        super().__init__()
        self.audioEncoder = inAudioEncoder
        self.textDecoder = inTextDecoder

    def forward(self, x: Tensor):
        audio_features = self.audioEncoder(x)

        n_layer_cross_k_list = []
        n_layer_cross_v_list = []
        for block in self.textDecoder.blocks:
            n_layer_cross_k_list.append(block.cross_attn.key(audio_features))
            n_layer_cross_v_list.append(block.cross_attn.value(audio_features))

        return torch.stack(n_layer_cross_k_list), torch.stack(n_layer_cross_v_list)

class MultiHeadAttentionCross(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.multiHeadAttention.query(x)
        wv, qk = self.multiHeadAttention.qkv_attention(q, k, v, mask)
        return self.multiHeadAttention.out(wv)

class MultiHeadAttentionSelf(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(
        self,
        x: Tensor,       #(b, n_ctx      , n_state)
        k_cache: Tensor, #(b, n_ctx_cache, n_state)
        v_cache: Tensor, #(b, n_ctx_cache, n_state)
        mask: Tensor,
    ):
        q = self.multiHeadAttention.query(x) #(b, n_ctx, n_state)
        k = self.multiHeadAttention.key(x)   #(b, n_ctx, n_state)
        v = self.multiHeadAttention.value(x) #(b, n_ctx, n_state)

        k = torch.cat((k_cache, k), 1) #(b, n_ctx_cache + n_ctx, n_state)
        v = torch.cat((v_cache, v), 1) #(b, n_ctx_cache + n_ctx, n_state)
        
        wv, qk = self.multiHeadAttention.qkv_attention(q, k, v, mask)
        return self.multiHeadAttention.out(wv), k, v


class ResidualAttentionBlockTensorCache(nn.Module):
    def __init__(self, inResidualAttentionBlock: ResidualAttentionBlock):
        super().__init__()
        self.originalBlock = inResidualAttentionBlock
        self.attn = MultiHeadAttentionSelf(inResidualAttentionBlock.attn)
        self.cross_attn = MultiHeadAttentionCross(inResidualAttentionBlock.cross_attn) if inResidualAttentionBlock.cross_attn else None

    def forward(
        self,
        x: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        mask: Tensor,
    ):
        self_attn_x, self_k_cache_updated, self_v_cache_updated = self.attn(self.originalBlock.attn_ln(x), self_k_cache, self_v_cache, mask=mask) 
        x = x + self_attn_x

        if self.cross_attn:
            x = x + self.cross_attn(self.originalBlock.cross_attn_ln(x), cross_k, cross_v)

        x = x + self.originalBlock.mlp(self.originalBlock.mlp_ln(x))
        return x, self_k_cache_updated, self_v_cache_updated

class TextDecoderTensorCache(nn.Module):
    def __init__(self, inTextDecoder: TextDecoder, in_n_ctx: int):
        super().__init__()
        self.textDecoder = inTextDecoder
        self.n_ctx = in_n_ctx

        self.blocks = []
        for orginal_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlockTensorCache(orginal_block))

    def forward(self, x: Tensor,
                n_layer_self_k_cache: Tensor,
                n_layer_self_v_cache: Tensor,
                n_layer_cross_k: Tensor,
                n_layer_cross_v: Tensor,
                offset: Tensor,
                ):
        x = self.textDecoder.token_embedding(x) + self.textDecoder.positional_embedding[offset[0] : offset[0] + x.shape[-1]]
        x = x.to(n_layer_cross_k[0].dtype)

        i = 0
        self_k_cache_list = []
        self_v_cache_list = []
        for block in self.blocks:
            x, self_k_cache, self_v_cache = block(x, 
                                                self_k_cache = n_layer_self_k_cache[i],
                                                self_v_cache = n_layer_self_v_cache[i],
                                                cross_k = n_layer_cross_k[i],
                                                cross_v = n_layer_cross_v[i],
                                                mask=self.textDecoder.mask)
            self_k_cache_list.append(self_k_cache)
            self_v_cache_list.append(self_v_cache)
            i += 1

        n_layer_self_k_cache = torch.stack(self_k_cache_list)
        n_layer_self_v_cache = torch.stack(self_v_cache_list)

        x = self.textDecoder.ln(x)

        logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits, n_layer_self_k_cache, n_layer_self_v_cache


model = whisper.load_model("base")

audio = whisper.load_audio(r"R:a.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

encoder = AudioEncoderTensorCache(model.encoder, model.decoder)
torch.onnx.export(
    encoder,
    mel.unsqueeze(0),
    "encoder.onnx",
    verbose=True,
    input_names=['mel'],
    output_names=['n_layer_cross_k', 'n_layer_cross_v'])

n_layer_cross_k, n_layer_cross_v = encoder(mel.unsqueeze(0))
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
n_audio = mel.shape[0]
tokens = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]

decoder = TextDecoderTensorCache(model.decoder, model.dims.n_text_ctx)
n_layer_self_k_cache = torch.zeros((len(model.decoder.blocks), model.dims.n_mels, 0, model.dims.n_text_state)).to(mel.device)
n_layer_self_v_cache = torch.zeros((len(model.decoder.blocks), model.dims.n_mels, 0, model.dims.n_text_state)).to(mel.device)
offset = torch.zeros(1, dtype=torch.int64).to(mel.device)
torch.onnx.export(
    decoder,
    (tokens, n_layer_self_k_cache, n_layer_self_v_cache, n_layer_cross_k, n_layer_cross_v, offset),
    "decoder.onnx",
    verbose=True,
    input_names=['tokens', 'in_n_layer_self_k_cache', 'in_n_layer_self_v_cache', 'n_layer_cross_k', 'n_layer_cross_v', 'offset'],
    output_names=['logits', 'out_n_layer_self_k_cache', 'out_n_layer_self_v_cache'],
    dynamic_axes={
                'tokens' : {1 : 'n_tokens'},
                'in_n_layer_self_k_cache' : {2 : 'n_tokens'},
                'in_n_layer_self_v_cache' : {2 : 'n_tokens'},
                })

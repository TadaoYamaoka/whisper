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

        k_cache[:,-k.shape[1]:,:] = k #(b, n_ctx_cache + n_ctx, n_state)
        v_cache[:,-v.shape[1]:,:] = v #(b, n_ctx_cache + n_ctx, n_state)
        
        wv, qk = self.multiHeadAttention.qkv_attention(q, k_cache, v_cache, mask)
        return self.multiHeadAttention.out(wv), k_cache, v_cache


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

    def forward(self, tokens: Tensor,
                n_layer_self_k_cache: Tensor,
                n_layer_self_v_cache: Tensor,
                n_layer_cross_k: Tensor,
                n_layer_cross_v: Tensor,
                offset: Tensor,
                ):
        x = self.textDecoder.token_embedding(tokens) + self.textDecoder.positional_embedding[offset[0] : offset[0] + tokens.shape[-1]]
        x = x.to(n_layer_cross_k[0].dtype)

        i = 0
        for block in self.blocks:
            self_k_cache = n_layer_self_k_cache[i,:,:offset[0] + tokens.shape[-1],:]
            self_v_cache = n_layer_self_v_cache[i,:,:offset[0] + tokens.shape[-1],:]
            x, self_k_cache, self_v_cache = block(x,
                                                  self_k_cache = self_k_cache,
                                                  self_v_cache = self_v_cache,
                                                  cross_k = n_layer_cross_k[i],
                                                  cross_v = n_layer_cross_v[i],
                                                  mask=self.textDecoder.mask)
            n_layer_self_k_cache[i,:,:offset[0] + tokens.shape[-1],:] = self_k_cache
            n_layer_self_v_cache[i,:,:offset[0] + tokens.shape[-1],:] = self_v_cache
            i += 1

        x = self.textDecoder.ln(x)

        logits = (x @ torch.transpose(self.textDecoder.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits, n_layer_self_k_cache, n_layer_self_v_cache


model = whisper.load_model("base")

audio = whisper.load_audio(r"D:\src\WhisperTest\a.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device).unsqueeze(0)

encoder = AudioEncoderTensorCache(model.encoder, model.decoder)
torch.onnx.export(
    encoder,
    mel,
    "encoder.onnx",
    verbose=True,
    input_names=['mel'],
    output_names=['n_layer_cross_k', 'n_layer_cross_v'])

n_layer_cross_k, n_layer_cross_v = encoder(mel)
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
n_audio = mel.shape[0]
tokens = torch.tensor([[tokenizer.sot, tokenizer.sot, tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 3]

decoder = TextDecoderTensorCache(model.decoder, model.dims.n_text_ctx)
# cacheは固定長
n_layer_self_k_cache = torch.zeros((len(model.decoder.blocks), n_audio, model.dims.n_text_ctx, model.dims.n_text_state), device=mel.device)
n_layer_self_v_cache = torch.zeros((len(model.decoder.blocks), n_audio, model.dims.n_text_ctx, model.dims.n_text_state), device=mel.device)
offset = torch.zeros(1, dtype=torch.int64).to(mel.device)

logits, n_layer_self_k_cache, n_layer_self_v_cache = decoder(tokens, n_layer_self_k_cache, n_layer_self_v_cache, n_layer_cross_k, n_layer_cross_v, offset)

offset = torch.tensor([tokens.shape[1]], dtype=torch.int64).to(mel.device)
tokens = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]

logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache = decoder(tokens, n_layer_self_k_cache, n_layer_self_v_cache, n_layer_cross_k, n_layer_cross_v, offset)

torch.onnx.export(
    decoder,
    (tokens, n_layer_self_k_cache, n_layer_self_v_cache, n_layer_cross_k, n_layer_cross_v, offset),
    "decoder.onnx",
    verbose=True,
    input_names=['tokens', 'in_n_layer_self_k_cache', 'in_n_layer_self_v_cache', 'n_layer_cross_k', 'n_layer_cross_v', 'offset'],
    output_names=['logits', 'out_n_layer_self_k_cache', 'out_n_layer_self_v_cache'],
    dynamic_axes={
                'tokens' : { 0: 'n_audio', 1 : 'n_tokens' },
                'in_n_layer_self_k_cache' : { 1: 'n_audio' },
                'in_n_layer_self_v_cache' : { 1: 'n_audio' },
                })

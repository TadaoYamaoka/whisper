import whisper
from openvino.runtime import Core
import torch
import numpy as np
import time

model = whisper.load_model("base", device='cpu')

audio = whisper.load_audio(r"../WhisperTest/a.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)

# Load the network to OpenVINO Runtime.
ie = Core()
model_encoder = ie.compile_model(model=ie.read_model(model='encoder.onnx'), device_name="CPU")

output_n_layer_cross_k = model_encoder.output(0)
output_n_layer_cross_v = model_encoder.output(1)

time_start = time.time()
output_encoder = model_encoder(mel.numpy())
n_layer_cross_k = output_encoder[output_n_layer_cross_k]
n_layer_cross_v = output_encoder[output_n_layer_cross_v]

print(time.time()- time_start)

print(n_layer_cross_k.shape)
print(n_layer_cross_v.shape)

tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
n_audio = mel.shape[0]
tokens = np.array([[tokenizer.sot, tokenizer.sot, tokenizer.sot]] * n_audio, np.int64)  # [n_audio, 3]
n_layer_self_k_cache = np.empty((len(model.decoder.blocks), n_audio, model.dims.n_text_ctx, model.dims.n_text_state), np.float32)
n_layer_self_v_cache = np.empty((len(model.decoder.blocks), n_audio, model.dims.n_text_ctx, model.dims.n_text_state), np.float32)
offset = np.zeros(1, dtype=np.int64)

model_decoder = ie.compile_model(model=ie.read_model(model='decoder.onnx'), device_name="CPU")

output_logits = model_decoder.output(0)
output_n_layer_self_k_cache = model_decoder.output(1)
output_n_layer_self_v_cache = model_decoder.output(2)

time_start = time.time()
output_decoder = model_decoder([tokens, n_layer_self_k_cache, n_layer_self_v_cache, n_layer_cross_k, n_layer_cross_v, offset])

print(time.time()- time_start)

logits = output_decoder[output_logits]
n_layer_self_k_cache = output_decoder[output_n_layer_self_k_cache]
n_layer_self_v_cache = output_decoder[output_n_layer_self_v_cache]

print(logits.shape)
print(n_layer_self_k_cache.shape)
print(n_layer_self_v_cache.shape)

offset = np.array([tokens.shape[1]], dtype=np.int64)
tokens = np.array([[tokenizer.sot]] * n_audio, np.int64)  # [n_audio, 1]
time_start = time.time()
output_decoder = model_decoder([tokens, n_layer_self_k_cache, n_layer_self_v_cache, n_layer_cross_k, n_layer_cross_v, offset])

print(time.time()- time_start)

logits = output_decoder[output_logits]
n_layer_self_k_cache = output_decoder[output_n_layer_self_k_cache]
n_layer_self_v_cache = output_decoder[output_n_layer_self_v_cache]

print(logits.shape)
print(n_layer_self_k_cache.shape)
print(n_layer_self_v_cache.shape)

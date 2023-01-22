import whisper
import torch
import onnxruntime
import numpy as np

model = whisper.load_model("base", device='cpu')

audio = whisper.load_audio(r"R:a.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).unsqueeze(0)

encoder_session = onnxruntime.InferenceSession('encoder.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

encoder_io_binding = encoder_session.io_binding()
encoder_io_binding.bind_cpu_input('mel', mel.numpy())
encoder_io_binding.bind_output('n_layer_cross_k', device_type='cuda')
encoder_io_binding.bind_output('n_layer_cross_v', device_type='cuda')

encoder_session.run_with_iobinding(encoder_io_binding)

n_layer_cross_k, n_layer_cross_v = encoder_io_binding.get_outputs()

print(n_layer_cross_k.shape())
print(n_layer_cross_v.shape())

tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
n_audio = mel.shape[0]
tokens = torch.tensor([[tokenizer.sot, tokenizer.sot, tokenizer.sot]] * n_audio)  # [n_audio, 3]
n_layer_self_k_cache = onnxruntime.OrtValue.ortvalue_from_shape_and_type((len(model.decoder.blocks), n_audio, model.dims.n_text_ctx, model.dims.n_text_state), element_type=np.float32, device_type='cuda', device_id=0)
n_layer_self_v_cache = onnxruntime.OrtValue.ortvalue_from_shape_and_type((len(model.decoder.blocks), n_audio, model.dims.n_text_ctx, model.dims.n_text_state), element_type=np.float32, device_type='cuda', device_id=0)
offset = np.zeros(1, dtype=np.int64)

decoder_session = onnxruntime.InferenceSession('decoder.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
decoder_io_binding = decoder_session.io_binding()
decoder_io_binding.bind_cpu_input('tokens', tokens.numpy())
decoder_io_binding.bind_ortvalue_input('in_n_layer_self_k_cache', n_layer_self_k_cache)
decoder_io_binding.bind_ortvalue_input('in_n_layer_self_v_cache', n_layer_self_v_cache)
decoder_io_binding.bind_ortvalue_input('n_layer_cross_k', n_layer_cross_k)
decoder_io_binding.bind_ortvalue_input('n_layer_cross_v', n_layer_cross_v)
decoder_io_binding.bind_cpu_input('offset', offset)
decoder_io_binding.bind_output('logits')
decoder_io_binding.bind_output('out_n_layer_self_k_cache')
decoder_io_binding.bind_output('out_n_layer_self_v_cache')

decoder_session.run_with_iobinding(decoder_io_binding)

logits, n_layer_self_k_cache, n_layer_self_v_cache = decoder_io_binding.get_outputs()

print(logits.shape())
print(n_layer_self_k_cache.shape())
print(n_layer_self_v_cache.shape())

offset = np.array([tokens.shape[1]], dtype=np.int64)
tokens = torch.tensor([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
decoder_io_binding = decoder_session.io_binding()
decoder_io_binding.bind_cpu_input('tokens', tokens.numpy())
decoder_io_binding.bind_ortvalue_input('in_n_layer_self_k_cache', n_layer_self_k_cache)
decoder_io_binding.bind_ortvalue_input('in_n_layer_self_v_cache', n_layer_self_v_cache)
decoder_io_binding.bind_ortvalue_input('n_layer_cross_k', n_layer_cross_k)
decoder_io_binding.bind_ortvalue_input('n_layer_cross_v', n_layer_cross_v)
decoder_io_binding.bind_cpu_input('offset', offset)
decoder_io_binding.bind_output('logits')
decoder_io_binding.bind_output('out_n_layer_self_k_cache')
decoder_io_binding.bind_output('out_n_layer_self_v_cache')

decoder_session.run_with_iobinding(decoder_io_binding)

logits, n_layer_self_k_cache, n_layer_self_v_cache = decoder_io_binding.get_outputs()

print(logits.shape())
print(n_layer_self_k_cache.shape())
print(n_layer_self_v_cache.shape())

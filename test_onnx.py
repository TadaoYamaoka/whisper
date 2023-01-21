import whisper
import torch
import onnxruntime

model = whisper.load_model("base", device='cpu')

audio = whisper.load_audio(r"R:a.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio)

encoder_session = onnxruntime.InferenceSession('encoder.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

encoder_io_binding = encoder_session.io_binding()
encoder_io_binding.bind_cpu_input('mel', mel.unsqueeze(0).numpy())
encoder_io_binding.bind_output('n_layer_cross_k', device_type='cuda')
encoder_io_binding.bind_output('n_layer_cross_v', device_type='cuda')

encoder_session.run_with_iobinding(encoder_io_binding)

n_layer_cross_k, n_layer_cross_v = encoder_io_binding.get_outputs()

print(n_layer_cross_k.numpy().shape)
print(n_layer_cross_v.numpy().shape)

tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
n_audio = mel.shape[0]
tokens = torch.tensor([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
n_layer_self_k_cache = torch.zeros((len(model.decoder.blocks), model.dims.n_mels, 0, model.dims.n_text_state))
n_layer_self_v_cache = torch.zeros((len(model.decoder.blocks), model.dims.n_mels, 0, model.dims.n_text_state))
offset = torch.zeros(1, dtype=torch.int64)

decoder_session = onnxruntime.InferenceSession('decoder.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
decoder_io_binding = decoder_session.io_binding()
decoder_io_binding.bind_cpu_input('tokens', tokens.numpy())
decoder_io_binding.bind_cpu_input('in_n_layer_self_k_cache', n_layer_self_k_cache.numpy())
decoder_io_binding.bind_cpu_input('in_n_layer_self_v_cache', n_layer_self_v_cache.numpy())
decoder_io_binding.bind_ortvalue_input('n_layer_cross_k', n_layer_cross_k)
decoder_io_binding.bind_ortvalue_input('n_layer_cross_v', n_layer_cross_v)
decoder_io_binding.bind_cpu_input('offset', offset.numpy())
decoder_io_binding.bind_output('logits')
decoder_io_binding.bind_output('out_n_layer_self_k_cache')
decoder_io_binding.bind_output('out_n_layer_self_v_cache')

decoder_session.run_with_iobinding(decoder_io_binding)

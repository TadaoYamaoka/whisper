import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('model_int8')
args = parser.parse_args()

quantized_model = quantize_dynamic(
    args.model,
    args.model_int8,
    weight_type=QuantType.QUInt8,
)

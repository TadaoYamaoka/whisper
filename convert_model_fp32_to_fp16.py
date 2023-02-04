import argparse
import onnx
from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('model_fp16')
parser.add_argument('--save_as_external_data', action='store_true')
args = parser.parse_args()

new_onnx_model = convert_float_to_float16_model_path(args.model)
onnx.save(new_onnx_model, args.model_fp16, save_as_external_data=args.save_as_external_data)

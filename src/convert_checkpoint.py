import argparse

import onnx
import torch

from lightning_module import SegmentModule

DEVICE = 'cpu'
ONNX_MODEL_NAME = './models/onnx_model/onnx_model_2.onnx'


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--model_path', type=str, default=ONNX_MODEL_NAME, help='Path to the *pt file')
    parser.add_argument('--device', type=str, default=DEVICE, help='specify device')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()

    device = torch.device(args.device)
    checkpoint_name = args.checkpoint  # Set checkpoint_name from command-line argument
    core_model = SegmentModule.load_from_checkpoint(checkpoint_name, map_location=device)._model  # noqa: WPS437
    core_model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)  # noqa: WPS432
    torch.onnx.export(
        core_model,
        dummy_input,
        args.model_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': [0], 'output': [0]},
    )

    onnx_model = onnx.load(args.model_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))  # noqa: WPS421

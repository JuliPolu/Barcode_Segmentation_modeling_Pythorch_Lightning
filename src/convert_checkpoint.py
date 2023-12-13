import torch
from lightning_module import SegmentModule
import argparse 
import onnx


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--model_path', type=str, default='./models/onnx_model/onnx_model.onnx', help='Path to the *pt file')
    parser.add_argument('--device', type=str, default='cpu', help='specify device')
    return parser.parse_args()
    

if __name__ == '__main__':

    args = arg_parse()
    checkpoint_name = args.checkpoint  # Set checkpoint_name from command-line argument
    core_model = SegmentModule.load_from_checkpoint(checkpoint_name, map_location=torch.device(args.device))._model
    core_model.eval()  

    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        core_model,
        dummy_input,
        args.model_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes = {'input': [0], 'output': [0]}, 
)

    onnx_model = onnx.load(args.model_path)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))
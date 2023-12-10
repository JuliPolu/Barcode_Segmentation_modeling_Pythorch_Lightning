import typing as tp
import numpy as np
import torch
from lightning_module import SegmentModule
import argparse 


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--model_path', type=str, default='../model_weights/final_model2.pt', help='Path to the *pt file')
    parser.add_argument('--device', type=str, default='cpu', help='specify device')
    return parser.parse_args()

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, size, encoder, threshold):
        super().__init__()
        self.model = model
        self.size = size
        self.threshold = threshold
        self.encoder = encoder
    
    def forward(self, x):
        return self.model(x)
    

if __name__ == '__main__':

    args = arg_parse()
    checkpoint_name = args.checkpoint  # Set checkpoint_name from command-line argument

    core_model = SegmentModule.load_from_checkpoint(checkpoint_name, map_location=torch.device(args.device))._model

    model_wrapper = ModelWrapper(core_model, size=256, encoder='resnet34', threshold=0.4)

    dummy_input = torch.randn(1, 3, 256, 256)
    traced_scripted_model = torch.jit.script(model_wrapper, dummy_input)
    torch.jit.save(traced_scripted_model, args.model_path)
    
    
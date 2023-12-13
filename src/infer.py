import typing as tp
import cv2
import numpy as np
import torch
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import argparse 
import segmentation_models_pytorch as smp
import onnxruntime as ort

MODEL_THRESHOLD = 0.4
TARGET_IMAGE_SIZE = (256, 256)
MODEL_ENCODER = 'resnet34'
PRETRAINED_WEIGHTS = 'imagenet'
ONNX_MODEL_NAME = './models/onnx_model/onnx_model.onnx'
IMAGE_PATH = "./data/images/000a8eff-08fb-4907-8b34-7a13ca7e37ea--ru.8e3b8a9a-9090-46ba-9c6c-36f5214c606d.jpg"

def mask_postprocesing(prob_mask: np.ndarray, threshold: float, original_size: tp.Tuple[int, int, int]) -> np.ndarray:
    mask = (prob_mask > threshold).astype(np.uint8)
    mask = mask.transpose(1, 2, 0)
    mask = cv2.resize(mask, (original_size), interpolation=cv2.INTER_NEAREST)
    num_labels, labels_im = cv2.connectedComponents(mask)
    
    max_size = 0
    max_label = 0
    for label in range(1, num_labels):
        component = (labels_im == label).astype(np.uint8)
        size = cv2.countNonZero(component)
        if size > max_size:
            max_size = size
            max_label = label

    largest_component_mask = (labels_im == max_label).astype(np.uint8)
    return largest_component_mask


def preprocess_image(image: np.ndarray, target_image_size: tp.Tuple[int, int]) -> torch.Tensor:
    image = image.astype(np.float32)
    processing_smp = smp.encoders.get_preprocessing_fn(MODEL_ENCODER, pretrained = PRETRAINED_WEIGHTS)
    preprocess = albu.Compose(
            [
                albu.Resize(height=target_image_size[0], width=target_image_size[1]),
                albu.Lambda(image=processing_smp),
                ToTensorV2(),
            ]
    )
    image_array = preprocess(image=image)['image'] 
    image_array = image_array.float().numpy() 
    return image_array

def mask_to_bbox(mask: np.ndarray) -> dict:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    bbox = {
        "bbox": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        }
    }
    return bbox

def run_inference(session, input_data):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    return outputs


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=ONNX_MODEL_NAME, help='Path to the checkpoint file')
    parser.add_argument('--image_path', type=str, default=IMAGE_PATH, help='Path to the image file')
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()
    image_path = args.image_path
    img = cv2.imread(image_path)
    original_size = (img.shape[1], img.shape[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img, TARGET_IMAGE_SIZE)
    
    ort_session = ort.InferenceSession(
        args.model_path,
        providers=['CPUExecutionProvider',],
    )

    output = run_inference(ort_session, img[None])
    prob_mask = torch.sigmoid(torch.tensor(output[0]))[0].numpy()
    mask = mask_postprocesing(prob_mask, MODEL_THRESHOLD, original_size)
    
    bbox = mask_to_bbox(mask)
    
    print(bbox)


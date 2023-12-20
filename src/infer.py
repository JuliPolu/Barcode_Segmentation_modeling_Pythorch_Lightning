import argparse
import typing as tp

import albumentations as albu
import cv2
import numpy as np
import onnxruntime as ort
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2

MODEL_THRESHOLD = 0.4
TARGET_IMAGE_SIZE = (256, 256)
MODEL_ENCODER = 'resnet34'
PRETRAINED_WEIGHTS = 'imagenet'
ONNX_MODEL_NAME = './models/onnx_model/onnx_model.onnx'
IMAGE_PATH = './data/images/000a8eff-08fb-4907-8b34-7a13ca7e37ea--ru.8e3b8a9a-9090-46ba-9c6c-36f5214c606d.jpg'


def mask_postprocesing(
    prob_mask: np.ndarray,
    threshold: float,
    original_size: tp.Tuple[int, int, int],
) -> np.ndarray:
    mask = create_binary_mask(prob_mask, threshold)
    mask = resize_mask(mask, original_size)
    return find_largest_connected_component(mask)


def create_binary_mask(prob_mask: np.ndarray, threshold: float) -> np.ndarray:
    return (prob_mask > threshold).astype(np.uint8)


def resize_mask(mask: np.ndarray, size: tp.Tuple[int, int]) -> np.ndarray:
    resized_mask = mask.transpose(1, 2, 0)
    return cv2.resize(resized_mask, size, interpolation=cv2.INTER_NEAREST)


def find_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels_im = cv2.connectedComponents(mask)

    max_label = find_label_of_largest_component(num_labels, labels_im)
    return (labels_im == max_label).astype(np.uint8)


def find_label_of_largest_component(num_labels: int, labels_im: np.ndarray) -> int:
    max_size = 0
    max_label = 0
    for label in range(1, num_labels):
        size = cv2.countNonZero((labels_im == label).astype(np.uint8))
        if size > max_size:
            max_size = size
            max_label = label

    return max_label


def preprocess_image(image: np.ndarray, target_image_size: tp.Tuple[int, int]) -> torch.Tensor:
    image = image.astype(np.float32)
    processing_smp = smp.encoders.get_preprocessing_fn(MODEL_ENCODER, pretrained=PRETRAINED_WEIGHTS)
    preprocess = albu.Compose(
        [
            albu.Resize(height=target_image_size[0], width=target_image_size[1]),
            albu.Lambda(image=processing_smp),
            ToTensorV2(),
        ],
    )
    image_array = preprocess(image=image)['image']
    return image_array.float().numpy()


def mask_to_bbox(mask: np.ndarray) -> dict:  # noqa: WPS210
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_indices = np.where(rows)[0]
    y_min = row_indices[0]
    y_max = row_indices[-1]
    col_indices = np.where(cols)[0]
    x_min = col_indices[0]
    x_max = col_indices[-1]

    return {
        'bbox': {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
        },
    }


def run_inference(session, input_data: torch.Tensor):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_data})


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
        providers=[
            'CPUExecutionProvider',
        ],
    )

    output = run_inference(ort_session, img[None])
    prob_mask = torch.sigmoid(torch.tensor(output[0]))[0]
    prob_mask = prob_mask.numpy()
    mask = mask_postprocesing(prob_mask, MODEL_THRESHOLD, original_size)

    bbox = mask_to_bbox(mask)

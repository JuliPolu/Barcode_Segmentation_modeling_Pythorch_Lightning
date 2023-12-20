import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch.utils.data import DataLoader

from src.augmentations import get_transforms
from src.config import Config
from src.dataset import SegmentationDataset
from src.lightning_module import SegmentModule

START_THRESHOLD = 0.2
END_THRESHOLD = 0.9
STEP_THRESHOLD = 0.05


def calculate_average_iou(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    thresholds: np.ndarray,
) -> dict:
    model.to(device)
    model.eval()
    iou_scores = {thresh: [] for thresh in thresholds}
    iou_metric = torchmetrics.JaccardIndex(num_classes=2, threshold=0.5, task='binary', average='macro')
    iou_metric = iou_metric.to(device)

    with torch.no_grad():
        process_dataloader(dataloader, model, thresholds, iou_metric, iou_scores)

    return {thresh: np.mean(scores) for thresh, scores in iou_scores.items()}


def process_dataloader(
    dataloader: DataLoader,
    model: torch.nn.Module,
    thresholds: np.ndarray,
    iou_metric: torchmetrics.JaccardIndex,
    iou_scores: dict,
) -> None:
    for images, true_masks in dataloader:
        images = images.float().to(device)
        true_masks = true_masks.float().to(device)
        preds = torch.sigmoid(model(images))
        update_iou_scores(preds, true_masks, thresholds, iou_metric, iou_scores)


def update_iou_scores(
    preds: torch.Tensor,
    true_masks: torch.Tensor,
    thresholds: np.ndarray,
    iou_metric: torchmetrics.JaccardIndex,
    iou_scores: dict,
) -> None:
    for thresh in thresholds:
        thresholded_preds = (preds > thresh).float()
        iou_metric(thresholded_preds, true_masks)
        score = iou_metric.compute().item()
        iou_scores[thresh].append(score)
        iou_metric.reset()


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    config = Config.from_yaml(args.config_file)

    # Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_directory = Path(config.data_config.data_path)
    checkpoint_name = args.checkpoint

    valid_transforms = get_transforms(
        width=config.data_config.width,
        height=config.data_config.height,
        encoder=config.data_config.encoder_name,
        augmentations=False,
    )
    valid_df = pd.read_csv(data_directory / 'df_valid.csv').to_numpy()
    valid_dataset = SegmentationDataset(
        valid_df,
        image_folder=data_directory,
        transforms=valid_transforms,
    )
    model = SegmentModule.load_from_checkpoint(checkpoint_name, map_location='cpu')
    dataloader = DataLoader(valid_dataset)
    thresholds = np.arange(START_THRESHOLD, END_THRESHOLD, STEP_THRESHOLD)

    avg_iou_scores = calculate_average_iou(model, dataloader, device, thresholds)
    best_threshold = max(avg_iou_scores, key=avg_iou_scores.get)

    output_message = f'Best threshold for IoU: {best_threshold} with score: {avg_iou_scores[best_threshold]:.4f}'

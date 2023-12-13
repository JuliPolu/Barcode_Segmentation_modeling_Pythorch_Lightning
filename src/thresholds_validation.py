import torch
import torchmetrics
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from augmentations import get_transforms
from dataset import SegmentationDataset
from lightning_module import SegmentModule


def calculate_average_iou(model, dataloader, device, thresholds):
    model.to(device)
    model.eval()
    iou_scores = {thresh: [] for thresh in thresholds}
    iou_metric = torchmetrics.JaccardIndex(num_classes=2, threshold=0.5, task='binary', average='macro')
    iou_metric = iou_metric.to(device)

    with torch.no_grad():
        for images, true_masks in dataloader:
            images, true_masks = images.float(), true_masks.float()
            images = images.to(device)
            true_masks = true_masks.to(device)
            preds = torch.sigmoid(model(images))
            for thresh in thresholds:
                thresholded_preds = (preds > thresh).float()
                iou_metric(thresholded_preds, true_masks)
                score = iou_metric.compute().item()
                iou_scores[thresh].append(score)
                iou_metric.reset()

    avg_iou_scores = {thresh: np.mean(iou_scores[thresh]) for thresh in thresholds}
    return avg_iou_scores

#Initialization
data_directory = "C:/Users/julia/CV/01-SEGMENTATION/data"
checkpoint_name = '../experiments/FPN_r34_d_f_256/epoch_epoch=14-val_IoU=0.892.ckpt'

valid_transforms = get_transforms(width=256, height=256, encoder = 'resnet34', augmentations=False)
valid_df = pd.read_csv(os.path.normpath(os.path.join(data_directory, 'df_train.csv')))
valid_dataset = SegmentationDataset(
    valid_df,
    image_folder=data_directory,
    transforms=valid_transforms,
)
model = SegmentModule.load_from_checkpoint(checkpoint_name)
dataloader = DataLoader(valid_dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
thresholds = np.arange(0.2, 0.9, 0.05)

avg_iou_scores = calculate_average_iou(model, dataloader, device, thresholds)
best_threshold = max(avg_iou_scores, key=avg_iou_scores.get)

print(f"Best threshold for IoU: {best_threshold} with score: {avg_iou_scores[best_threshold]}")
from torchmetrics import F1Score, JaccardIndex, MetricCollection


def get_metrics(**kwargs) -> MetricCollection:
    # Assuming your task is binary segmentation (barcode vs background)
    # For multi-class segmentation, adjust `num_classes` and `average` accordingly
    return MetricCollection(
        {
            'f1': F1Score(**kwargs),
            'IoU': JaccardIndex(**kwargs),
        },
    )

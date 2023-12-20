# Комментарии и история экспериментов

### Архитектура

- Unet
- FPN - показал наилучший результат

### Бэкбоны

Пробовала:

- resnet34  - итоговый выбор
- timm-efficientnet-b0
- timm-efficientnet-b3 - получилось на 1 сотую выше, чем resnet34, однако учится значительно дольше - поэтому итоговый выбор пал на resnet34

### Pretrain

Пробовала:

- imagenet - показал наилучший результат
- noisy-student

### Optimizers and learning rate

- Adam lr: 1e-3 weight_decay: 1e-5
- scheduler: 'CosineAnnealingLR'
- scheduler: 'ReduceLROnPlateau' - показал наилучший результат

### Loss function

Пробовала:

- Lovasz
- Dice
- Dice(0.7), Focal(0.3)
- Dice(0.6), Focal(0.4) - наилучший результат

### Метрики

- F1Score
- JaccardIndex -  по нему осуществлялся мониторинг

### Аугментации

- albu.ShiftScaleRotate
- albu.OneOf(\[albu.CLAHE, albu.RandomBrightnessContrast\])

### Threshold

- тренировала на отсечке 0.5, затем кроссвалидировала на валидационной выборке
- подбор отсечек осуществляется в [thresholds_validation.py](src/thresholds_validation.py)

### Image size

- 224 x 224
- 256 x 256 - наилучший результат
- 320 x 320

### Открытые вопросы

1. Не смогла конвертировать с использованием torchscript (пробовала и script и trace - постоянно ошибки, гугление не помогло, видимо проблема со сложной архитектурой моделей из segmentation_models.pytorch). Подскажите какие-нибудь лайфхаки для конвертации torchscript сложных моделек
1. В итоге конвертировала в onnx и инферила через onnx_runtime. Из минусов вижу, что в модель не получается завернуть дополнительные параметры типа threshold и размера изображения для модели
1. Очень прошу дать комментарии по выбранным параметрам модели и лучшим практикам

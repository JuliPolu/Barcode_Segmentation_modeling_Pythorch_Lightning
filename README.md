# Barcode Segmentation Model

## Основная задача

Создание модели сегментация штрих-кодов по фотографиям c последующим распознаванием цифр на каждом штрих-коде (OCR).

### Датасет

Трейн датасет включает 540 фото, предварительно собранных с помощью Tолоки.

[Ссылка](https://disk.yandex.ru/d/pRFNuxLQUZcDDg) на исходный датасет

Первичный анализ и подготовка данных в папке [тетрадке](notebooks/EDA.ipynb)

### Обучение

Запуск тренировки:

```
PYTHONPATH=. python src/train.py configs/FPN_r34_d_f_256.yaml.yaml
```

### Логи финальной модели в ClearML

Перформанс модели можно посмотреть тут:

[ClearML](https://app.clear.ml/projects/03d0e1fbd7854729b147d47e858fcc91/experiments/a59da8e539e645f0ab47fb7f0a5a6b21/output/execution)


### Актуальная версия чекпойнта модели:

dvc pull models/model_checkpoint/epoch_epoch=05-val_IoU=0.895.ckpt.dvc

### Актуальная версия сохраненной torscript модели:

dvc pull models/onnx_model/onnx_model.onnx.dvc

### Инеренс

Посмотреть результаты работы обученной сети можно посмотреть в [тетрадке](notebooks/inference.ipynb)

А также запустить скрипт для конвертации чекпойнта в onnx
```
PYTHONPATH=. python src/convert_checkpoint.py --checkpoint ./models/model_checkpoint/epoch_epoch=05-val_IoU=0.895.ckpt --model_path ./models/onnx_model/onnx_model.onnx
```

И запустить скрипт для инференса
```
PYTHONPATH=.  python ./src/infer.py --model_path ./models/onnx_model/onnx_model.onnx --image_path ./data/images/000a8eff-08fb-4907-8b34-7a13ca7e37ea--ru.8e3b8a9a-9090-46ba-9c6c-36f5214c606d.jpg
```

### Комментарии и история экспериментов 

Кратко подбор параметров и оставшиеся вопросы описаны в файле [HISTORY&COMMENTS.md](HISTORY&COMMENTS.md)

Очень бы хотела получить максимально развернутые коментарии по корректности используемых параметров и вообще по лучшим практикам, так как опыта пока маловато, и любой комментарий от экспертов на вес золота! 
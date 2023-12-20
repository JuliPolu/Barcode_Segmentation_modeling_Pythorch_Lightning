.PHONY: *

VENV=venv
PYTHON=$(VENV)/bin/python3
DEVICE=gpu
DATASET_FOLDER := data


# ================== LOCAL WORKSPACE SETUP ==================
venv:
	python3 -m venv $(VENV)
	@echo 'Path to Python executable $(shell pwd)/$(PYTHON)'


pre_commit_install:
	@echo "=== Installing pre-commit ==="
	$(PYTHON) -m pre_commit install


install_all: venv
	@echo "=== Installing common dependencies ==="
	$(PYTHON) -m pip install -r requirements.txt

	make pre_commit_install


fetch_dataset_from_yadisk:
	# Download dataset to local folder
	# Alternative option to get dataset if you don't have access to DVC
	wget "https://disk.yandex.ru/d/pRFNuxLQUZcDDg" -O $(DATASET_FOLDER)/data_segmentation.zip
	unzip -q $(DATASET_FOLDER)/data_segmentation.zip -d $(DATASET_FOLDER)
	rm $(DATASET_FOLDER)/data_segmentation.zip
	find $(DATASET_FOLDER) -type f -name '.DS_Store' -delete


# ========================= TRAINING ========================
run_training:
	$(PYTHON) -m  src.train configs/FPN_r34_d_f_256.yaml

threshold_validation:
	$(PYTHON) -m  src.thresholds_validation.py --config_file /path/to/config --checkpoint /path/to/checkpoint


# ============================ DVC ==========================
dvc_checkpoint:
	dvc pull models/model_checkpoint/epoch_epoch=05-val_IoU=0.895.ckpt.dvc


dvc_onnx:
	dvc pull models/onnx_model/onnx_model.onnx.dvc


# ========================= INFERENCE ========================
convert_checkpoint:

	$(PYTHON) src/convert_checkpoint.py --checkpoint ./models/model_checkpoint/epoch_epoch=05-val_IoU=0.895.ckpt

inference:
	$(PYTHON)  src/infer.py --model_path ./models/onnx_model/onnx_model.onnx --image_path ./data/images/000a8eff-08fb-4907-8b34-7a13ca7e37ea--ru.8e3b8a9a-9090-46ba-9c6c-36f5214c606d.jpg


# ================== CONTINUOUS INTEGRATION =================
ci_test:
	$(PYTHON) -m pytest tests

ci_static_code_analysis:
	$(PYTHON) -m pre_commit run --all-files
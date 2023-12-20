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
	$(PYTHON) -m  src.train /path/to/configs/

threshold_validation:
	$(PYTHON) -m  src.thresholds_validation.py --config_file /path/to/config --checkpoint /path/to/checkpoint


# ============================ DVC ==========================
dvc_checkpoint:
	dvc pull /path/to/checkpoint.dvc


dvc_onnx:
	dvc pull /path/to/model.dvc


# ========================= INFERENCE ========================
convert_checkpoint:

	$(PYTHON) src/convert_checkpoint.py --checkpoint /path/to/checkpoint

inference:
	$(PYTHON)  src/infer.py --model_path /path/to/model --image_path /path/to/image

# ================== CONTINUOUS INTEGRATION =================
ci_test:
	$(PYTHON) -m pytest tests

ci_static_code_analysis:
	$(PYTHON) -m pre_commit run --all-files
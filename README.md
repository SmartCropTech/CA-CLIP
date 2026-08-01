# CA-CLIP and CCPLD

This repository provides the three reproducibility materials requested for the
study:

1. **CCPLD dataset**: `CCPLD/`
2. **Prompt bank**: `prompt_bank/disease_prompt_bank.json`
3. **Configuration and model files**: `configuration/`

## Repository structure

```text
CA-CLIP/
|-- CCPLD/
|   |-- train/
|   `-- val/
|-- prompt_bank/
|   `-- disease_prompt_bank.json
|-- configuration/
|   |-- config.json
|   |-- multi_disease_construction.py
|   |-- ca_clip_framework.py
|   |-- ca_clip_model.py
|   `-- mobilenet_v3_distilled.py
|-- weight/
|   |-- mobilenet_v3.pth
|   `-- best_thresholds.npy
|-- predict.py
|-- requirements.txt
`-- LICENSE
```

## File descriptions

- `CCPLD/` contains 3,400 co-occurring disease leaf images: 2,380 training
  images and 1,020 validation images across 17 disease combinations.
- `disease_prompt_bank.json` contains the structured class-specific prompt
  bank. CA-CLIP uses the P3 `visible_symptom` field.
- `config.json` records the dataset, CA-CLIP, prompt, threshold, and distilled
  MobileNetV3 settings used by the released code.
- `multi_disease_construction.py` is the phenotype-prior multi-disease image
  construction implementation, using Apple combinations as the documented
  crop-specific example.
- `ca_clip_framework.py` contains the architecture-only CA-CLIP definition.
- `ca_clip_model.py` exposes the CA-Gating and CA-CLIP architecture classes.
- `mobilenet_v3_distilled.py` contains the distilled MobileNetV3-Large model,
  checkpoint loading, class-specific thresholds, and inference implementation.
- `predict.py` is the user entry point for either one image or an image folder.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.11 and PyTorch 2.3.0 were used for the reported experiments.

## Image or folder prediction

Open `predict.py` and set:

```python
INPUT_PATH = r"path/to/image.jpg"
```

or:

```python
INPUT_PATH = r"path/to/image_folder"
```

Then run `predict.py` directly in an IDE. Predictions are saved to
`predictions.csv`. The released `best_thresholds.npy` provides one
validation-optimized decision threshold per class.

Command-line overrides remain optional:

```bash
python predict.py --input path/to/images --output predictions.csv
```

## CA-CLIP

`configuration/ca_clip_framework.py` presents the CLIP image encoder,
residual adapter, prompt-conditioned modulation, class-aware gate, and
multi-label classification head. It is an architecture reference and does not
include the training pipeline.

## License

The source code is provided under the MIT License. Dataset use must also comply
with the source-image licenses described in the accompanying manuscript.

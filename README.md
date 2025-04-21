# hyundai_competition_no2

This repository implements an end-to-end OCR pipeline to recognize alphanumeric text from images using a TPS‑ResNet‑BiLSTM‑Attention model with data augmentation.

## Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- nltk
- scikit-learn
- torch.utils.tensorboard (TensorBoard)
- tqdm
- Pillow

Install dependencies:
```bash
pip install torch torchvision nltk scikit-learn tensorboard tqdm pillow
```

## Repository Structure

```
project_root/
├── code/                    # (optional helper scripts)
│   ├── augmentation/        # data augmentations
│   │   ├── blur.py          # DefocusBlur, MotionBlur
│   │   └── noise.py         # GaussianNoise, ShotNoise
│   ├── model/               # modules for sequence modeling
│   │   ├── feature_extraction.py
│   │   ├── prediction.py
│   │   ├── sequence_modeling.py
│   │   └── transformation.py
├── config.py                # hyperparameter definitions (args)
├── converter.py             # text‑to‑index and index‑to‑text converter
├── dataset.py               # Custom Dataset & collate functions
├── ocrmodel.py              # OCRModel definition (TPS‑ResNet‑BiLSTM‑Attn)
├── train.py                 # Main training script
├── result.py                # Inference and result export script
├── datasets/                # data folders
│   ├── train/
│   └── test/
├── weights/                 # (optional) pre‑trained weights
└── result.csv               # final predictions on the test set
```

## Data Preparation

Place your data under the `datasets/` directory:

```
datasets/
├── train/
│   ├── <class_folder>/      # e.g. alphanumeric categories or source batches
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── annotations.txt  # tab‑separated lines: <filename>\t<label>
└── test/
    ├── <class_folder>/      # same structure without labels (optional)
    │   ├── imageA.jpg
    │   └── imageB.jpg
```

The training script (`train.py`) will scan each `train/<folder>/annotations.txt`, read lines of the form:
```
image_filename\tLABEL_STRING
```
and load corresponding images.

## Configuration

Edit `config.py` to customize hyperparameters. Typical entries include:
```python
args.gpu             = 0
args.save_model_name = "experiment1"
args.epochs          = 200
args.batch           = 64
args.lr              = 1.0
args.rho             = 0.95
args.eps             = 1e-8
args.seed            = 7777
args.text_max_length = 25
```

Or adjust values in this file before running.

## Training

To train the model, simply run:
```bash
python train.py
```

- **Checkpoints** are saved under `save/<save_model_name>/` (created automatically).
- **Logs** for TensorBoard are written to `log/<save_model_name>/` (created automatically).

Launch TensorBoard to monitor:
```bash
tensorboard --logdir log
```

## Inference

After training, generate predictions on your test set by running:
```bash
python result.py --gpu 0 --load_model save/<save_model_name>/<checkpoint>.pkl
```

Outputs will be aggregated into `result.csv`.

## Notes

- The `train.py` script applies random blur and noise augmentations for robustness.
- Model performance is evaluated using accuracy and normalized edit distance (CER), logged per epoch.

## Acknowledgments

- Based on the TPS‑ResNet‑BiLSTM‑Attention OCR architecture.
- Uses NLTK’s edit distance for error rate computation.
- Data augmentations inspired by common OCR pipelines.

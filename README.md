# hyundai_competition_no2

This repository implements an end-to-end OCR pipeline to recognize alphanumeric text from images using a TPS‑ResNet‑BiLSTM‑Attention model with data augmentation.

## ❗️ Project Summary

---

1. **진행기간:** 2022.01 ~ 2022.02
2. **팀명: 현대중공업떡상기원** (한주혁, 차준영, 김민성)
3. **역할:** 프로젝트 리더, 외부 데이터 수집 및 새로운 데이터 샘플 생성
4. **기술스택: `Python`**, **`PyTorch`**, **`timm`** 
5. **결과 및 성과:** 
    - 최종보고서 [**[📄]**](https://drive.google.com/file/d/1A59tbe-t8AxI-XemlvMK8jSnxUAGaznz/view?usp=drive_link)
    - https://github.com/hannn0403/hyundai_competition_no2
    - [현대중공업그룹] 제 2회 조선/해양산업 디지털 혁신을 위한 Big Data / AI 대학생 경진대회 장려상 수상.
6. **주요내용:** 강재 문자 인식 과제에서는 HandWritten Data 7,732개와 Printing Data 300개로 인한 극심한 데이터 불균형과 HandWritten Data에서 글자 굵기가 지나치게 두꺼워 식별이 어려운 문제를 해결하기 위해, EMINST와 Chars 74K Data를 추가 수집하여 두 데이터셋을 보완한 후, Morphological Transformation과 Unsharp Masking 기법을 활용하여 이미지 샤프닝 처리를 진행하였다. 이후 TPS Spatial Transformation, ResNet Feature Extraction, Bidirectional LSTM, Attention Prediction으로 구성된 OCR 파이프라인을 적용하여 모델을 학습 및 예측하였고, 79.21%의 Accuracy와 95.55%의 1-NED를 달성하였다.

---

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

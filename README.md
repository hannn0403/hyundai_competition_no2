# hyundai_competition_no2

### requirements

<img src="https://img.shields.io/badge/python-3.8-yellowgreen"/> 
<img src="https://img.shields.io/badge/nltk-3.7-yellowgreen"/> 
<img src="https://img.shields.io/badge/numpy-1.21.2-yellowgreen"/> 
<img src="https://img.shields.io/badge/opencv-4.1.2-yellowgreen"/> 
<img src="https://img.shields.io/badge/pandas-1.4.0-red"/> 
<img src="https://img.shields.io/badge/pillow-8.4.0-red"/> 
<img src="https://img.shields.io/badge/pytorch-1.10.2-blue"/>
<img src="https://img.shields.io/badge/skimage-0.19.1-yellowgreen"/> 
<img src="https://img.shields.io/badge/sklearn-1.0.2-yellowgreen"/> 
<img src="https://img.shields.io/badge/torchvision-0.11.3-yellowgreen"/> 
<img src="https://img.shields.io/badge/torchsummary-1.5.1-yellowgreen"/> 
<img src="https://img.shields.io/badge/tqdm-4.62.3-yellowgreen"/> 
<img src="https://img.shields.io/badge/wand-0.6.7-yellowgreen"/> 



########## directory ##########
└ code
   └ augmentation
      └ blur.py		Blurring Image
      └ noise.py		Add noise to Image
   └ model
      └ feature_extraction.py	
      └ prediction.py		
      └ sequence_modeling.py	
      └ transformation.py	
   └ config.py		Configuration File
   └ converter.py		Convert between text-label and text-index
   └ dataset.py		Load dataset
   └ ocrmodel.py		TPS Spatial Transformer-ResNet Feature Extractor-Bidirectonal LSTM-Attention Predictor
   └ result.py		★ Make Submission File
   └ train.py		Train / Validate Model
└ datasets
   └ test
   └ train
└ weights
   └ 
└ result.csv		★ Submission File

########## Getting Start ##########
python result.py

## Directory

project_root/
├── code/
│   ├── augmentation/
│   │   ├── blur.py                  # Blurring Image
│   │   └── noise.py                 # Add noise to Image
│   ├── model/
│   │   ├── feature_extraction.py    # 
│   │   ├── prediction.py            #
│   │   ├── sequence_modeling.py     #
│   │   └── transformation.py        #
│   ├── config.py                    # Configuration File
│   ├── converter.py                 # Convert between text-label and text-index
│   ├── dataset.py                   # Load dataset
│   ├── ocrmodel.py                  # TPS Spatial Transformer-ResNet Feature Extractor-Bidirectional LSTM-Attention Predictor
│   ├── result.py                    # Make Submission File
│   └── train.py                     # Train / Validate Model
├── datasets/
│   ├── test/
│   └── train/
├── weights/
└── result.csv                       # Submission File




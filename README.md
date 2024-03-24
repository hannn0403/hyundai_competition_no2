# hyundai_competition_no2


########## requirements ##########
python			3.8
nltk			3.7
numpy			1.21.2
opencv-contrib-python	4.1.2.30
pandas			1.4.0
pillow 			8.4.0
pytorch			1.10.2
skimage			0.19.1
sklearn			1.0.2
torchvision		0.11.3
torchsummary		1.5.1
tqdm			4.62.3
wand			0.6.7

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




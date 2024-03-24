import argparse
parser = argparse.ArgumentParser(description="Hyundai Heavy Industries Competition No.2")


# Train or Test
parser.add_argument("--fine_tuning", default=True,
                    help="Use pretrained model by ClovaAI")
# GPU
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=7777)

# Preprocessing
parser.add_argument("--img_height", type=int, default=32)
parser.add_argument("--img_width", type=int, default=100)

# Model Network
parser.add_argument("--SequenceModeling", default=True,
                    help="SequenceModeling stage. None | BiLSTM")
parser.add_argument("--text_max_length", type=int, default=15,
                    help="maximum-label-length")
parser.add_argument("--num_fiducial", type=int, default=20,
                    help="number of fiducial points of TPS-STN")
parser.add_argument("--input_channel", type=int, default=1,
                    help="the number of input channel of Feature extractor")
parser.add_argument("--output_channel", type=int, default=512,
                    help="the number of output channel of Feature extractor")
parser.add_argument("--hidden_size", type=int, default=256,
                    help="the size fo the LSTM hidden stage")

# Model Training
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--split", type=float, default=0.3)
parser.add_argument("--save_model_name", type=str, default="experiment-final-2")
parser.add_argument("--load_model_name", type=str, default="experiment-final-2_197_best_norm_ED.pkl")

# Optimizer
parser.add_argument('--lr', type=float, default=1.0,
                    help='default=1.0 for Adadelta')
parser.add_argument('--rho', type=float, default=0.95,
                    help='decay rate rho for Adadelta. default=0.95')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='eps for Adadelta. default=1e-8')


# args
args = parser.parse_args()

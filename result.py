import os
import glob
import pandas as pd
import tqdm
import time
from nltk.metrics.distance import edit_distance
import torch
from torch.utils.data import DataLoader
from config import args
from ocrmodel import OCRModel
from converter import AttnLabelConverter
from dataset import CompNo2Dataset, AlignCollate


def test():
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    path = f"../datasets/test"

    # Load Dataset (Training / Validation)
    print("Load Test Dataset")
    x, y = [], []
    for folder in os.listdir(path):
        text_file = glob.glob(f"{path}/{folder}/*.txt")[0]
        with open(text_file, "r") as f:
            while True:
                line = f.readline().strip().split("\t")
                if len(line) == 1:
                    break
                elif not (line[1] == "X" or line[1] == "x"):
                    x.append(f"{folder}/{line[0]}")
                    y.append(line[1])

    collate = AlignCollate(args)
    test_dataset = CompNo2Dataset(args, path, x, y)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate)

    # Convert Text Label to Index
    converter = AttnLabelConverter("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZt-.")
    args.num_class = len(converter.character)

    # Load Model
    print("Model: TPS-ResNet-BiLSTM-Attn")
    model = OCRModel(args).to(device)
    print(f"Load Pretrained Model: experiment-final-2_197_best_norm_ED")
    model.load_state_dict(torch.load("../weights/experiment-final-2_197_best_norm_ED.pkl")["model_state_dict"])

    # Test
    print(f"\nPrediction")
    image_list = []
    label_list = []
    predicted_list = []
    norm_ED_list = []

    start_time = time.time()
    for batch_item in tqdm.tqdm(test_dataloader):
        model.eval()

        # load image / label
        images = batch_item["image"].to(device)
        labels = batch_item["label"]
        labels_index, length = converter.encode(labels, args.text_max_length)
        labels_index = labels_index.to(device)
        batch_size = images.size(0)

        # for max length prediction
        length_for_predict = torch.IntTensor([args.text_max_length] * batch_size).to(device)
        text_for_predict = torch.LongTensor(batch_size, args.text_max_length + 1).fill_(0).to(device)

        # test / decoding
        with torch.no_grad():
            preds_index = model(images, text_for_predict, is_train=False)
            preds_index = preds_index[:, :labels_index.shape[1] - 1, :]

            _, preds_index = preds_index.max(2)
            preds_str = converter.decode(preds_index, length_for_predict)
            labels_str = converter.decode(labels_index[:, 1:], length)

        # scoring
        image_list += batch_item["file"]
        label_list += batch_item["label"]
        for gt, pred in zip(labels_str, preds_str):
            gt = gt[:gt.find("[s]")]
            pred_EOS = pred.find("[s]")
            pred = pred[:pred_EOS]
            if len(gt) == 0 or len(pred) == 0:
                norm_ED = 0
            elif len(gt) > len(pred):
                norm_ED = 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED = 1 - edit_distance(pred, gt) / len(pred)

            predicted_list.append(pred)
            norm_ED_list.append(norm_ED)
    end_time = time.time()

    result_dict = {
        "Image list": [i.split("/")[1] for i in image_list],
        "GT": label_list,
        "Prediction": predicted_list,
        "accuracy": [f"{norm_ED * 100:0.2f}%" for norm_ED in norm_ED_list],
        "Average Accuracy": sum(norm_ED_list) / len(norm_ED_list),
        "Inference Speed (ms)": f"{(end_time - start_time) / len(image_list):0.5f}ms/image"
    }
    result_csv = pd.DataFrame(result_dict)
    result_csv.to_csv("../result.csv", index=False)
    print("────────────────── T E S T   C O M P L E T E !! ──────────────────")


if __name__ == "__main__":
    test()

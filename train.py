import os
import glob
import tqdm
from nltk.metrics.distance import edit_distance
import torch.cuda
from torch import nn
from torch import optim
from torch.nn import init
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from config import args
from ocrmodel import OCRModel
from converter import AttnLabelConverter
from dataset import CompNo2Dataset, AlignCollate
from augmentation.blur import DefocusBlur, MotionBlur
from augmentation.noise import GaussianNoise, ShotNoise


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train():
    writer = SummaryWriter(f"../log/{args.save_model_name}")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    path = f"../datasets/train"
    if not os.path.exists(f'../save/{args.save_model_name}'):
        os.makedirs(f'../save/{args.save_model_name}')

    # Load Dataset (Training / Validation)
    print("Load Training / Validation Dataset")
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7777)
    collate = AlignCollate(args)

    train_transform = transforms.Compose([
        GaussianNoise(),
        ShotNoise(),
        DefocusBlur(),
        MotionBlur()
    ])

    train_dataset = CompNo2Dataset(args, path, x_train, y_train, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate)
    val_dataset = CompNo2Dataset(args, path, x_test, y_test)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate)

    # Convert Text Label to Index
    converter = AttnLabelConverter("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZt-.")
    args.num_class = len(converter.character)

    # Load Model
    print("Model: TPS-ResNet-BiLSTM-Attn")
    model = OCRModel(args).to(device)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # Set Criterion, Optimizer
    print("Loss function: Cross Entropy")
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore ["GO"] token (index 0)
    print("Optimizer: Adadelta")
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=args.rho, eps=args.eps)

    # Epoch
    print("\nStart Training:", args.save_model_name)
    best_accuracy, best_norm_ED = -1, -1
    best_accuracy_epoch, best_norm_ED_epoch = -1, -1
    for epoch in range(1, args.epochs+1):

        """ Training """
        train_loss, train_item = 0, 0
        length_of_data, n_correct, norm_ED = 0, 0, 0
        tqdm_train_dataloader = tqdm.tqdm(enumerate(train_dataloader))
        for idx, batch_item in tqdm_train_dataloader:
            model.train()
            optimizer.zero_grad()

            # load image / label
            images = batch_item["image"].to(device)
            labels = batch_item["label"]
            labels_index, length = converter.encode(labels, args.text_max_length)
            labels_index = labels_index.to(device)
            batch_size = images.size(0)
            length_of_data = length_of_data + batch_size
            length_for_predict = torch.IntTensor([args.text_max_length] * batch_size).to(device)

            # training
            preds_index = model(images, labels_index[:, :-1])
            target_index = labels_index[:, 1:]

            # loss
            loss = criterion(preds_index.view(-1, preds_index.shape[-1]), target_index.contiguous().view(-1))
            train_loss += loss.data.sum()
            train_item += loss.data.numel()
            loss.backward()

            # optimization
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # decoding
            _, preds_index = preds_index.max(2)
            preds_str = converter.decode(preds_index, length_for_predict)
            labels_str = converter.decode(labels_index[:, 1:], length)

            # scoring
            for gt, pred in zip(labels_str, preds_str):
                gt = gt[:gt.find("[s]")]
                pred_EOS = pred.find("[s]")
                pred = pred[:pred_EOS]

                if gt == pred:
                    n_correct += 1

                # Normalized Edit Distance
                if len(gt) == 0 or len(pred) == 0:
                    norm_ED += 0
                elif len(gt) > len(pred):
                    norm_ED += 1 - edit_distance(pred, gt) / len(gt)
                else:
                    norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # log
            tqdm_train_dataloader.set_postfix({
                "LR": optimizer.param_groups[0]["lr"],
                "Epoch": f"[{epoch}/{args.epochs}]",
                "Batch": f"[{idx + 1}/{len(train_dataloader)}]",
                "Train Loss": "{:06f}".format(train_loss / train_item),
                "Accuracy": n_correct / float(length_of_data) * 100,
                "Norm_ED": norm_ED / float(length_of_data)
            })
        total_train_loss = train_loss / train_item
        train_accuracy = n_correct / float(length_of_data) * 100
        train_norm_ED = norm_ED / float(length_of_data)

        """ Validation """
        val_loss, val_item = 0, 0
        length_of_data, n_correct, norm_ED = 0, 0, 0
        tqdm_val_dataloader = tqdm.tqdm(enumerate(val_dataloader))
        with torch.no_grad():
            for idx, batch_item in tqdm_val_dataloader:
                model.eval()

                # load image / label
                images = batch_item["image"].to(device)
                labels = batch_item["label"]
                labels_index, length = converter.encode(labels, args.text_max_length)
                labels_index = labels_index.to(device)
                batch_size = images.size(0)
                length_of_data = length_of_data + batch_size
                length_for_predict = torch.IntTensor([args.text_max_length] * batch_size).to(device)
                text_for_predict = torch.LongTensor(batch_size, args.text_max_length + 1).fill_(0).to(device)

                # test
                preds_index = model(images, text_for_predict, is_train=False)
                preds_index = preds_index[:, :labels_index.shape[1]-1, :]
                target_index = labels_index[:, 1:]

                # loss
                loss = criterion(preds_index.contiguous().view(-1, preds_index.shape[-1]), target_index.contiguous().view(-1))
                val_loss += loss.data.sum()
                val_item += loss.data.numel()

                # decoding
                _, preds_index = preds_index.max(2)
                preds_str = converter.decode(preds_index, length_for_predict)
                labels_str = converter.decode(labels_index[:, 1:], length)

                # scoring
                for gt, pred in zip(labels_str, preds_str):
                    gt = gt[:gt.find("[s]")]
                    pred_EOS = pred.find("[s]")
                    pred = pred[:pred_EOS]

                    if gt == pred:
                        n_correct += 1

                    # Normalized Edit Distance
                    if len(gt) == 0 or len(pred) == 0:
                        norm_ED += 0
                    elif len(gt) > len(pred):
                        norm_ED += 1 - edit_distance(pred, gt) / len(gt)
                    else:
                        norm_ED += 1 - edit_distance(pred, gt) / len(pred)

                # log
                tqdm_val_dataloader.set_postfix({
                    "LR": optimizer.param_groups[0]["lr"],
                    "Epoch": f"[{epoch}/{args.epochs}]",
                    "Batch": f"[{idx + 1}/{len(val_dataloader)}]",
                    "Validation Loss": "{:06f}".format(val_loss / val_item),
                    "Accuracy": n_correct / float(length_of_data) * 100,
                    "Norm_ED": norm_ED / float(length_of_data)
                })
        total_val_loss = val_loss / val_item
        val_accuracy = n_correct / float(length_of_data) * 100
        val_norm_ED = norm_ED / float(length_of_data)

        print(f"{'Ground Truth':25s} | {'Prediction':25s} | {'File Name'}")
        for gt, pred, file in zip(labels_str[:5], preds_str[:5], batch_item["file"]):
            print(f"{gt[:gt.find('[s]')]:25s} | {pred[:pred.find('[s]')]:25s} | {file}")
        print(f"{'Train Loss':17s}: {total_train_loss:0.5f}, {'Train Acc':17s}: {train_accuracy:0.5f}, {'Train Norm ED':17s}: {train_norm_ED:0.5f}")
        print(f"{'Val Loss':17s}: {total_val_loss:0.5f}, {'Val Acc':17s}: {val_accuracy:0.5f}, {'Val NormED':17s}: {val_norm_ED:0.5f}")
        writer.add_scalars("Loss", {"Train": total_train_loss, "Validation": total_val_loss}, epoch)
        writer.add_scalars("Accuracy", {"Train": train_accuracy, "Validation": val_accuracy}, epoch)
        writer.add_scalars("Norm ED", {"Train": train_norm_ED, "Validation": val_norm_ED}, epoch)

        if val_norm_ED > best_norm_ED and epoch >= 150:
            best_norm_ED = val_norm_ED
            best_norm_ED_epoch = epoch
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(state, f"../save/{args.save_model_name}/{args.save_model_name}_{epoch}_best_norm_ED.pkl")

        if epoch % 25 == 0:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(state, f"../save/{args.save_model_name}/{args.save_model_name}_{epoch}.pkl")
        print("─" * 80)

    print("────────────────── T R A I N   C O M P L E T E !! ──────────────────")
    print(f"Best Accuray: {best_accuracy_epoch}, {best_accuracy}")
    print(f"Best Norm ED: {best_norm_ED_epoch}, {best_norm_ED}")


if __name__ == "__main__":
    train()

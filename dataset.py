import cv2
import math
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


# Sharpening Image
def unsharp_masking(img, k):
    gaussian_filter_I = cv2.GaussianBlur(img, (7, 7), 0)
    g_mask = np.array(img, dtype=np.float32) - np.array(gaussian_filter_I, dtype=np.float32)
    g_mask = np.clip(g_mask, 0, 255)
    g_mask = g_mask.astype('uint8')
    g = np.array(img, dtype=np.float32) + k * g_mask
    g = np.clip(g, 0, 255)
    result = g.astype('uint8')
    return result


class CompNo2Dataset(Dataset):
    def __init__(self, config, path, files, labels, transform=None):
        self.config = config
        self.path = path
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        path = f"{self.path}/{self.files[item]}"

        # load image
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # image sharpening, Erosion
        folder = self.files[item].split("/")[0]
        if folder == "01" or folder == "02" or folder == "06":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            image = cv2.erode(image, kernel)
            image = unsharp_masking(image, 3)

        # cv2 to PIL Image
        image = Image.fromarray(image)

        # transform
        if self.transform is not None:
            image = self.transform(image)

        return {
            "file": self.files[item],
            "image": image,
            "label": self.labels[item]
        }


# Padding with Normalization
class NormalizePAD:
    def __init__(self, input_channel, height, width):
        self.toTensor = transforms.ToTensor()
        self.max_size = (input_channel, height, width)
        self.max_width_half = math.floor(width / 2)

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return Pad_img


# Align Collate (Fit the image size to (32, 100))
class AlignCollate:
    def __init__(self, config):
        self.config = config
        self.imgH = self.config.img_height
        self.imgW = self.config.img_width

    def __call__(self, batches):
        files, images, labels = [], [], []
        for batch in batches:
            files.append(batch["file"])
            images.append(batch["image"])
            labels.append(batch["label"])

        # same concept with 'Rosetta' paper
        resized_max_w = self.imgW
        input_channel = 3 if images[0].mode == 'RGB' else 1
        transform = NormalizePAD(input_channel, self.imgH, resized_max_w)

        # resizing iamge
        resized_images = []
        for image in images:
            w, h = image.size
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)
            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)

            # padding
            resized_images.append(transform(resized_image))

        # make tensor
        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        return {
            "file": files,
            "image": image_tensors,
            "label": labels
        }

import os
import glob
import torch
import numpy


def modify_textfile():
    text_file = f"../datasets/train/03/gt_train_03.txt"
    text_file_rev = f"../datasets/train/03/gt_train_03_rev.txt"

    x, y = [], []
    with open(text_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            line = line.strip().split("\t")
            x.append(line[0].split("/")[1])
            y.append(line[1])

    with open(text_file_rev, "w") as f:
        for image, label in zip(x, y):
            f.write(f"{image}\t{label}\n")


def operator():
    array = numpy.array([[0, 1, 2, 3, 4],
                         [5, 6, 7, 8, 9]])
    # print(array[:, :-1])
    # print(array.shape[-1])
    print(torch.IntTensor([20] * 16))


if __name__ == "__main__":
    operator()

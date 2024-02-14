# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import os
import math

import csv

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import InterpolationMode

def main():
    transform = transforms.Compose(
        [
            transforms.Resize(
                int(math.ceil(224 / 0.95)), interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    #train_dataset = ImageFolder(root="/dataset/imagenet/train", transform=transform)
    #labels, my_map = train_dataset.find_classes("/dataset/imagenet/train")
    train_dataset = ImageFolder(root="/dataset/imagenet/val", transform=transform)
    labels, my_map = train_dataset.find_classes("/dataset/imagenet/val")
    print(f"my_map = {my_map}")

    with open('class_to_idx_val.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in my_map.items():
            writer.writerow([key, value])

if __name__ == "__main__":
    main()

import random

import cv2
import numpy as np
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)

    def __call__(self, image, label):
        image = image.astype(np.float32) / 255.0
        label = np.ceil(label.astype(np.float32) / 255.0)
        return (image - self.mean) / self.std, label


class Scale:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image, label):
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return image, label


class RandomCropResize:
    def __init__(self, crop_area):
        self.crop_area = crop_area

    def __call__(self, image, label):
        if random.random() >= 0.5:
            return image, label
        height, width = image.shape[:2]
        x = random.randint(0, self.crop_area)
        y = random.randint(0, self.crop_area)
        image = cv2.resize(image[y:height-y, x:width-x], (width, height), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label[y:height-y, x:width-x], (width, height), interpolation=cv2.INTER_NEAREST)
        return image, label


class RandomFlip:
    def __call__(self, image, label):
        if random.random() < 0.5:
            image, label = cv2.flip(image, 0), cv2.flip(label, 0)
        if random.random() < 0.5:
            image, label = cv2.flip(image, 1), cv2.flip(label, 1)
        return image, label


class RandomExchange:
    def __call__(self, image, label):
        if random.random() < 0.5:
            image = np.concatenate((image[:, :, 3:6], image[:, :, 0:3]), axis=2)
        return image, label


class ToTensor:
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, image, label):
        if self.scale != 1:
            height, width = label.shape[:2]
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (width // self.scale, height // self.scale), interpolation=cv2.INTER_NEAREST)
        image = image[:, :, ::-1].copy().transpose(2, 0, 1)
        return torch.from_numpy(image), torch.from_numpy(np.asarray(label, dtype=np.int64).copy()).unsqueeze(0)

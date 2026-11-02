import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, dataset, file_root, transform=None, image_dir_a='A',
                 image_dir_b='B', label_dir='label', list_path=None):
        self.root = os.path.abspath(file_root)
        self.split = dataset
        self.transform = transform
        self.image_dir_a = image_dir_a
        self.image_dir_b = image_dir_b
        self.label_dir = label_dir

        candidates = []
        if list_path:
            candidates.append(list_path)
        candidates.extend([
            os.path.join(self.root, 'list', dataset + '.txt'),
            os.path.join(self.root, dataset + '.txt'),
            os.path.join(os.path.dirname(self.root), dataset + '.txt'),
        ])
        selected_list = next((path for path in candidates if path and os.path.isfile(path)), None)
        if selected_list:
            with open(selected_list, encoding='utf-8') as names_file:
                self.file_list = [line.strip().split()[0] for line in names_file if line.strip()]
        else:
            label_root = os.path.join(self.root, self.label_dir)
            self.file_list = sorted(
                name for name in os.listdir(label_root)
                if os.path.isfile(os.path.join(label_root, name)) and not name.startswith('.')
            )

        if not self.file_list:
            raise RuntimeError(f'No samples found for split {dataset} under {self.root}')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        path_a = os.path.join(self.root, self.image_dir_a, name)
        path_b = os.path.join(self.root, self.image_dir_b, name)
        label_name = name.replace('tif', 'png') if name.endswith('tif') else name
        label_path = os.path.join(self.root, self.label_dir, label_name)

        image_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
        image_b = cv2.imread(path_b, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if image_a is None or image_b is None or label is None:
            raise FileNotFoundError(f'Failed to read paired sample {name}: {path_a}, {path_b}, {label_path}')

        image = np.concatenate((image_a, image_b), axis=2)
        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1).copy())
            label = torch.from_numpy(label.copy()).unsqueeze(0)
        return image, label

    def get_img_info(self, idx):
        image = cv2.imread(os.path.join(self.root, self.image_dir_a, self.file_list[idx]))
        if image is None:
            raise FileNotFoundError(self.file_list[idx])
        return {'height': image.shape[0], 'width': image.shape[1]}

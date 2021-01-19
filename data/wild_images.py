import random
import os
from os.path import join as ospj

import torch
import torch.utils.data as data
from PIL import Image


class Wild(data.Dataset):
    def __init__(self, data_root, dataset_name, mode, transform_img, selected_attrs):
        super().__init__()
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.mode = mode
        self.selected_attrs = selected_attrs

        self.image_dir = ospj(self.data_root, dataset_name, 'images')
        self.attr_file = ospj(self.data_root, dataset_name, 'wild_pred_attributes_list.txt')
        self.transform_img = transform_img

        self.attr2idx = {}
        self.idx2attr = {}

        # self.test_images = []
        self.test_dataset = []

        self.preprocess()
        self.num_images = len(self.test_dataset)

    def preprocess(self):
        assert os.path.exists(self.image_dir), f'Image data directory does not exist: {self.image_dir}'
        assert os.path.exists(self.attr_file), f'Image attribute file does not exist: {self.attr_file}'
        with open(self.attr_file, 'r') as f:
            img_name_attrs_lines = f.readlines()
        all_attr_names = img_name_attrs_lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = img_name_attrs_lines[2:]
        for i, line in enumerate(lines):
            split = line.strip().split()
            filename = split[0]
            values = split[1:]
            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            filepath = ospj(self.image_dir, filename)
            self.test_dataset.append([filepath, label])
            # self.test_images.append(filepath)

        print(f'Finished preprocessing the {self.dataset_name} dataset...')

    def __getitem__(self, index):
        if self.mode == 'test':
            filepath, label = self.test_dataset[index]
            image = Image.open(filepath).convert('RGB')
        else:
            image = None
            raise NotImplementedError
        return {
            "real_img": self.transform_img(image), "real_cond": torch.FloatTensor(label),
            'sample_id': index, 'real_img_path': filepath,
        }

    def __len__(self):
        return self.num_images

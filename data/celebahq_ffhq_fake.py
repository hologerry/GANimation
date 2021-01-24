import copy
import os
import random
from os.path import join as ospj

import torch
import torch.utils.data as data
from PIL import Image


class CelebAHQFFHQFake(data.Dataset):
    def __init__(self, data_root, dataset_name, mode, transform_img, selected_attrs):
        super().__init__()
        self.data_root = data_root
        self.celeba_dataset_name = 'celebahq'
        self.ffhq_dataset_name = 'ffhq'
        self.mode = mode
        self.selected_attrs = selected_attrs
        self.transform_img = transform_img

        self.celeba_image_dir = ospj(self.data_root, self.celeba_dataset_name, 'CelebA-HQ-img')
        self.celeba_attr_file = ospj(self.data_root, self.celeba_dataset_name, 'CelebAMask-HQ-attribute-anno.txt')
        if 'Skin_0' in self.selected_attrs:
            self.celeba_attr_file = ospj(self.data_root, self.celeba_dataset_name, 'CelebAMask-HQ-attribute-anno-skin.txt')
        self.ffhq_image_dir = ospj(self.data_root, self.ffhq_dataset_name, 'images1024x1024')
        self.ffhq_attr_file = ospj(self.data_root, self.ffhq_dataset_name, 'ffhq_attributes_list.txt')

        self.attr2idx = {}
        self.idx2attr = {}
        self.celeba_train_dataset = []
        self.celeba_test_dataset = []
        self.ffhq_train_dataset = []
        self.ffhq_test_dataset = []

        self.celeba_preprocess()
        self.ffhq_preprocess()

        if self.mode == 'eval_part1':
            self.celeba_test_dataset = self.celeba_test_dataset[:len(self.celeba_test_dataset)//2]
            self.ffhq_test_dataset = self.ffhq_test_dataset[:len(self.ffhq_test_dataset)//2]
        elif self.mode == 'eval_part2':
            self.celeba_test_dataset = self.celeba_test_dataset[len(self.celeba_test_dataset)//2:]
            self.ffhq_test_dataset = self.ffhq_test_dataset[len(self.ffhq_test_dataset)//2:]

        if self.mode == 'train':
            self.num_image_attr_pairs = len(self.celeba_train_dataset) + len(self.ffhq_train_dataset)
        else:
            self.num_image_attr_pairs = len(self.celeba_test_dataset) + len(self.ffhq_test_dataset)
        self.train_dataset = self.celeba_train_dataset + self.ffhq_train_dataset
        self.test_dataset = self.celeba_test_dataset + self.ffhq_test_dataset

        if self.mode == 'ffhq_test':
            self.test_dataset = self.ffhq_test_dataset[:len(self.ffhq_test_dataset)//2]  # part 1
            self.num_image_attr_pairs = len(self.test_dataset)

    def celeba_preprocess(self):
        assert os.path.exists(self.celeba_image_dir), f'Image data directory does not exist: {self.celeba_image_dir}'
        assert os.path.exists(self.celeba_attr_file), f'Attribute file does not exist: {self.celeba_attr_file}'
        with open(self.celeba_attr_file, 'r') as f:
            img_name_attrs_lines = f.readlines()
        all_attr_names = img_name_attrs_lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        lines = img_name_attrs_lines[2:]

        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.strip().split()
            filename = split[0]
            values = split[1:]
            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            filepath = ospj(self.celeba_image_dir, filename)
            if i < 2000:
                self.celeba_test_dataset.append([filepath, label])
            else:  # 28000
                self.celeba_train_dataset.append([filepath, label])
        print(f'Finished preprocessing the {self.celeba_dataset_name} dataset...')

    def ffhq_preprocess(self):
        assert os.path.exists(self.ffhq_image_dir), f'Image data directory does not exist: {self.ffhq_image_dir}'
        assert os.path.exists(self.ffhq_attr_file), f'Attribute file does not exist: {self.ffhq_attr_file}'
        with open(self.ffhq_attr_file, 'r') as f:
            img_name_attrs_lines = f.readlines()

        lines = img_name_attrs_lines[2:]
        for i, line in enumerate(lines):
            split = line.strip().split()
            filename = split[0]
            values = split[1:]
            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            img_sub_dir = f'{(i // 1000):02d}000'
            filepath = ospj(self.ffhq_image_dir, img_sub_dir, filename)
            if i >= 66000:
                self.ffhq_test_dataset.append([filepath, label])
            else:  # 4000
                self.ffhq_train_dataset.append([filepath, label])
        print(f'Finished preprocessing the {self.ffhq_dataset_name} dataset...')

    def __getitem__(self, index):
        if self.mode == 'train':
            filename_a, label_a = self.train_dataset[index]
            random_index_b = random.randint(0, len(self.train_dataset) - 1)
            filename_b, label_b = self.train_dataset[random_index_b]

            image_a = Image.open(filename_a)
            image_b = Image.open(filename_b)

            return {
                "real_img": self.transform_img(image_a), "real_cond": torch.FloatTensor(label_a),
                "desired_img": self.transform_img(image_b), "desired_cond": torch.FloatTensor(label_b),
                'sample_id': index, 'real_img_path': filename_a
            }

        elif self.mode == 'val':
            filename_a, label_a = self.test_dataset[index]
            random_index_b = random.randint(0, len(self.test_dataset) - 1)
            while random_index_b == index:
                random_index_b = random.randint(0, len(self.test_dataset) - 1)
            filename_b, label_b = self.test_dataset[random_index_b]
            image_a = Image.open(filename_a)
            image_b = Image.open(filename_b)
            return {
                "real_img": self.transform_img(image_a), "real_cond": torch.FloatTensor(label_a),
                "desired_img": self.transform_img(image_b), "desired_cond": torch.FloatTensor(label_b),
                'sample_id': index, 'real_img_path': filename_a
            }

        else:  # mode 'test', 'eval_part1', 'eval_part2'
            filename_a, label_a = self.test_dataset[index]
            image_a = Image.open(filename_a)

            return {
                "real_img": self.transform_img(image_a), "real_cond": torch.FloatTensor(label_a),
                'sample_id': index, 'real_img_path': filename_a
            }

    def __len__(self):
        return self.num_image_attr_pairs

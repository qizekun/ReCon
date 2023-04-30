import os
import torch
import numpy as np
import cv2
import pandas as pd
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
from utils.transforms import get_transforms


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.img_path = config.IMG_PATH
        self.ratio = config.ratio
        
        self.text_list = {}
        self.index_list = {}
        for index, row in pd.read_json(config.TEXT_PATH).iterrows():
            self.text_list["0" + str(row['catalogue'])] = row['describe']
            self.index_list["0" + str(row['catalogue'])] = index
        self.subset = config.subset
        self.npoints = config.N_POINTS

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines[: int(self.ratio * len(lines))]:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] load ratio is {self.ratio}', logger='ShapeNet-55')
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        pc = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        img = cv2.imread(os.path.join(self.img_path, sample['file_path'].replace(".npy", ".png")))
        img = get_transforms()['train'](img)
        pc = self.random_sample(pc, self.sample_points_num)
        pc = self.pc_norm(pc)
        pc = torch.from_numpy(pc).float()
        text = self.text_list[sample['taxonomy_id']]
        index = self.index_list[sample['taxonomy_id']]
        label = torch.tensor(index)
        return sample['taxonomy_id'], sample['model_id'], pc, img, text, label

    def __len__(self):
        return len(self.file_list)

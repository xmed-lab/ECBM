import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
# AWA2_DATA_DIR="YOUR/AWA2/DATA/DIR"
AWA2_DATA_DIR="/home/xxucb/data/pcbm_dataset/Animals_with_Attributes2"

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 50



SElECTED_CONCEPTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

CONCEPT_SEMANTICS = ['antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', 'persian+cat', 'horse', 'german+shepherd', 'blue+whale', 'siamese+cat', 'skunk', 'mole', 'tiger', 'hippopotamus', 'leopard', 'moose', 'spider+monkey', 'humpback+whale', 'elephant', 'gorilla', 'ox', 'fox', 'sheep', 'seal', 'chimpanzee', 'hamster', 'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'rat', 'weasel', 'otter', 'buffalo', 'zebra', 'giant+panda', 'deer', 'bobcat', 'pig', 'lion', 'mouse', 'polar+bear', 'collie', 'walrus', 'raccoon', 'cow', 'dolphin']

class AnimalDataset(Dataset):
  def __init__(self, data_path, transform):
    root_dir='/'.join(data_path.split('/')[:-1])
    predicate_binary_mat = np.array(np.genfromtxt(f'{root_dir}/predicate-matrix-binary.txt', dtype='int'))
    self.predicate_binary_mat = predicate_binary_mat
    self.transform = transform

    class_to_index = dict()
    # Build dictionary of indices to classes
    with open(f"{root_dir}/classes.txt") as f:
      index = 1
      for line in f:
        class_name = line.split('\t')[1].strip()
        class_to_index[class_name] = index
        index += 1
    self.class_to_index = class_to_index

    df = pd.read_csv(data_path)
    img_names = df['img_name'].tolist()
    img_index = df['img_index'].tolist()
   
    self.img_names = img_names
    self.img_index = img_index

  def __getitem__(self, index):
    im = Image.open(self.img_names[index])
    if im.getbands()[0] == 'L':
      im = im.convert('RGB')
    if self.transform:
      im = self.transform(im)
    # if im.shape != (3,224,224):
    #   print(self.img_names[index])
    im_index = self.img_index[index]-1
    im_predicate = self.predicate_binary_mat[im_index,:]
    return im, im_index, torch.FloatTensor(im_predicate)

  def __len__(self):
    return len(self.img_names)


def load_data(data_path, batch_size, shuffle=True, num_workers=4, resol=224, is_training=True):
    if is_training:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
        ])
    dataset = AnimalDataset(data_path, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def generate_data(config, root_dic=AWA2_DATA_DIR, seed=42, output_dataset_vars=False, resol=224, is_training=True):
    if  root_dic is None:
        root_dic = AWA2_DATA_DIR
    concept_group_map = None
    imbalance = None
    n_concepts = len(SElECTED_CONCEPTS)
    seed_everything(seed)
    
    train_dl = load_data(AWA2_DATA_DIR + '/train.csv', config.batch_size, shuffle=True, num_workers=8, resol=resol, is_training=True)
    
    val_dl = load_data(AWA2_DATA_DIR + '/val.csv', config.batch_size, shuffle=False, num_workers=8, resol=resol, is_training=False)
    
    test_dl = load_data(AWA2_DATA_DIR + '/test.csv', config.batch_size, shuffle=False, num_workers=8, resol=resol, is_training=False)
    
    return train_dl, val_dl, test_dl, imbalance, (n_concepts, N_CLASSES, concept_group_map)
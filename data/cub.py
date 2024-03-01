
"""
Obtained from https://github.com/yewsiang/ConceptBottleneck and modified.
General utils for training, evaluation and data loading
"""
import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader
import collections
from .cub_stats import N_ATTRIBUTES, CONCEPT_SEMANTICS, SELECTED_CONCEPTS


CONCEPT_GROUP_MAP = collections.defaultdict(list)
for i, concept_name in enumerate(list(np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS])):
    group = concept_name[:concept_name.find("::")]
    CONCEPT_GROUP_MAP[group].append(i)
def select_concept_group(n,n_concept):
    count=0
    to_be_intervened=[]
    for i in CONCEPT_GROUP_MAP:
        to_be_intervened+=CONCEPT_GROUP_MAP[i]
        count+=1
        if count==n:
            return to_be_intervened



class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, num_classes, transform=None, pkl_itself=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])        
        if pkl_itself is None:
            for file_path in pkl_file_paths:
                self.data.extend(pickle.load(open(file_path, 'rb')))
        else:
            self.data = pkl_itself
        self.transform = transform
        self.use_attr = use_attr
        # print(use_attr)
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths

        idx = img_path.split('/').index('CUB_200_2011')
        img_path = '/'.join([self.image_dir] + img_path.split('/')[idx+1:])
        img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
                # print(len(attr_label))
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(N_ATTRIBUTES), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, torch.tensor(attr_label)
        else:
            return img, class_label
        
        

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['class_label']#[0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples


def load_cub_data(pkl_paths, use_attr, no_img, batch_size, uncertain_label=False, n_class_attr=2, image_dir='images', resampling=False, resol=224,
             normalizer=transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2]),
             n_classes=200):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """
    is_training = any(['train.pkl' in f for f in pkl_paths])
    if is_training:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            normalizer
            ])
    else:
        transform = transforms.Compose([
            transforms.CenterCrop(resol),
            transforms.ToTensor(), 
            normalizer
            ])

    dataset = CUBDataset(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, n_classes, transform)

    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    if resampling:
        sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size, drop_last=drop_last)
        loader = DataLoader(dataset, batch_sampler=sampler,num_workers=4)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,num_workers=4)
    return loader

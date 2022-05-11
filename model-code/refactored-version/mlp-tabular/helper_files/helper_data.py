import random

import pandas as pd
import torch
from torch.utils.data import sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets


def label_to_levels(label, num_classes, dtype=torch.float32):
    """Converts integer class label to extended binary label vector

    Parameters
    ----------
    label : int
        Class label to be converted into a extended
        binary vector. Should be smaller than num_classes-1.

    num_classes : int
        The number of class clabels in the dataset. Assumes
        class labels start at 0. Determines the size of the
        output vector.

    dtype : torch data type (default=torch.float32)
        Data type of the torch output vector for the
        extended binary labels.

    Returns
    ----------
    levels : torch.tensor, shape=(num_classes-1,)
        Extended binary label vector. Type is determined
        by the `dtype` parameter.

    Examples
    ----------
    >>> label_to_levels(0, num_classes=5)
    tensor([0., 0., 0., 0.])
    >>> label_to_levels(1, num_classes=5)
    tensor([1., 0., 0., 0.])
    >>> label_to_levels(3, num_classes=5)
    tensor([1., 1., 1., 0.])
    >>> label_to_levels(4, num_classes=5)
    tensor([1., 1., 1., 1.])
    """
    if not label <= num_classes-1:
        raise ValueError('Class label must be smaller or '
                         'equal to %d (num_classes-1). Got %d.'
                         % (num_classes-1, label))
    if isinstance(label, torch.Tensor):
        int_label = label.item()
    else:
        int_label = label

    levels = [1]*int_label + [0]*(num_classes - 1 - int_label)
    levels = torch.tensor(levels, dtype=dtype)
    return levels


def levels_from_labelbatch(labels, num_classes, dtype=torch.float32):
    """
    Converts a list of integer class label to extended binary label vectors

    Parameters
    ----------
    labels : list or 1D orch.tensor, shape=(num_labels,)
        A list or 1D torch.tensor with integer class labels
        to be converted into extended binary label vectors.

    num_classes : int
        The number of class clabels in the dataset. Assumes
        class labels start at 0. Determines the size of the
        output vector.

    dtype : torch data type (default=torch.float32)
        Data type of the torch output vector for the
        extended binary labels.

    Returns
    ----------
    levels : torch.tensor, shape=(num_labels, num_classes-1)

    Examples
    ----------
    >>> levels_from_labelbatch(labels=[2, 1, 4], num_classes=5)
    tensor([[1., 1., 0., 0.],
            [1., 0., 0., 0.],
            [1., 1., 1., 1.]])
    """
    levels = []
    for label in labels:
        levels_from_label = label_to_levels(
            label=label, num_classes=num_classes, dtype=dtype)
        levels.append(levels_from_label)

    levels = torch.stack(levels)
    return levels


def proba_to_label(probas):
    """
    Converts predicted probabilities from extended binary format
    to integer class labels

    Parameters
    ----------
    probas : torch.tensor, shape(n_examples, n_labels)
        Torch tensor consisting of probabilities returned by CORAL model.

    Examples
    ----------
    >>> # 3 training examples, 6 classes
    >>> probas = torch.tensor([[0.934, 0.861, 0.323, 0.492, 0.295],
    ...                        [0.496, 0.485, 0.267, 0.124, 0.058],
    ...                        [0.985, 0.967, 0.920, 0.819, 0.506]])
    >>> proba_to_label(probas)
    tensor([2, 0, 5])
    """
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    # adopted from https://github.com/galatolofederico/pytorch-balanced-batch/blob/master/sampler.py
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Parameters:
        ------------
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
        ------------
        Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_dataloaders_mnist(batch_size, num_workers=0,
                          validation_fraction=None,
                          train_transforms=None,
                          test_transforms=None):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=train_transforms,
                                   download=True)

    valid_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=test_transforms)

    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * 60000)
        train_indices = torch.arange(0, 60000 - num)
        valid_indices = torch.arange(60000 - num, 60000)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader


def get_dataloaders_cifar10(batch_size, num_workers=0,
                            validation_fraction=None,
                            train_transforms=None,
                            test_transforms=None):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(root='data',
                                     train=True,
                                     transform=train_transforms,
                                     download=True)

    valid_dataset = datasets.CIFAR10(root='data',
                                     train=True,
                                     transform=test_transforms)

    test_dataset = datasets.CIFAR10(root='data',
                                    train=False,
                                    transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * 50000)
        train_indices = torch.arange(0, 50000 - num)
        valid_indices = torch.arange(50000 - num, 50000)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader


######################################################################################

class HotspotDataset_v1(Dataset):

    def __init__(self, csv_path):
    
        df = pd.read_csv(csv_path)
        self.y = torch.from_numpy(df['ddG'].values).to(torch.int64)
        df = df.drop('ddG', axis=1)
        self.features = torch.from_numpy(df.values).to(torch.float32)

    def __getitem__(self, index):
        features = self.features[index]
        label = self.y[index]
        return features, label

    def __len__(self):
        return self.y.shape[0]

    
    
def get_dataloaders_hotspot_v1(batch_size, train_csv_path, test_csv_path, balanced=False, num_workers=0):

    train_dataset = HotspotDataset_v1(csv_path=train_csv_path)
    test_dataset = HotspotDataset_v1(csv_path=test_csv_path)
    
    
    if balanced:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=False,
                                  sampler=BalancedBatchSampler(train_dataset, labels=train_dataset.y),
                                  num_workers=num_workers)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=num_workers)

        
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader


class HotspotDataset_v2_2class(Dataset):

    def __init__(self, csv_path):
    
        feature_list = ['avg bond number', 'Hbond', 
            'Hphob', 'consurf', "B' side chain", "hotspot ratio"]
        df = pd.read_csv(csv_path)
        self.y = torch.from_numpy(df['2-class'].values).to(torch.int64)
        self.features = torch.from_numpy(df[feature_list].values).to(torch.float32)

    def __getitem__(self, index):
        features = self.features[index]
        label = self.y[index]
        return features, label

    def __len__(self):
        return self.y.shape[0]


def get_dataloaders_hotspot_v2(batch_size, train_csv_path, test_csv_path, balanced=False, num_workers=0, num_classes=2):

    if num_classes == 2:
        train_dataset = HotspotDataset_v2_2class(csv_path=train_csv_path)
        test_dataset = HotspotDataset_v2_2class(csv_path=test_csv_path)
    elif num_classes == 3:
        raise NotImplementedError('Not implemented yet')
    else:
        raise ValueError('num_classes option invalid')        
    
    
    if balanced:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=False,
                                  sampler=BalancedBatchSampler(train_dataset, labels=train_dataset.y),
                                  num_workers=num_workers)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=num_workers)

        
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader


#############################################


class HotspotDataset_v3_2class(Dataset):

    def __init__(self, csv_path):
    
        feature_list = ['avg bond number', 'Hbond', 
            'Hphob', 'consurf', "B' side chain", "hotspot ratio"]
        
        df = pd.read_csv(csv_path)
        self.y = torch.from_numpy(df['2-class'].values).to(torch.int64)
        self.features = torch.from_numpy(df[feature_list].values).to(torch.float32)
        
        ## add One-hot encoded amino acids
        codes = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 
                 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        code_to_int = {c:i for i,c in enumerate(codes)}     
        df['residue'] = df['residue'].map(code_to_int)
        tensor = torch.from_numpy(df['residue'].values)
        onehot = torch.nn.functional.one_hot(tensor).to(torch.float32)
        self.features = torch.cat((self.features, onehot), dim=1)
        

    def __getitem__(self, index):
        features = self.features[index]
        label = self.y[index]
        return features, label

    def __len__(self):
        return self.y.shape[0]
    
    
def get_dataloaders_hotspot_v3(batch_size, train_csv_path, test_csv_path, balanced=False, num_workers=0, num_classes=2):

    if num_classes == 2:
        train_dataset = HotspotDataset_v3_2class(csv_path=train_csv_path)
        test_dataset = HotspotDataset_v3_2class(csv_path=test_csv_path)
    elif num_classes == 3:
        raise NotImplementedError('Not implemented yet')
    else:
        raise ValueError('num_classes option invalid')        
    
    
    if balanced:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=False,
                                  sampler=BalancedBatchSampler(train_dataset, labels=train_dataset.y),
                                  num_workers=num_workers)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=num_workers)

        
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader


#############################################



class HotspotDataset_v3_2_2class(Dataset):

    def __init__(self, csv_path):
    
        feature_list = ['avg bond number', 'Hbond', 
            'Hphob', 'consurf', "B' side chain"]
        
        df = pd.read_csv(csv_path)
        self.y = torch.from_numpy(df['2-class'].values).to(torch.int64)
        self.features = torch.from_numpy(df[feature_list].values).to(torch.float32)
        
        ## add One-hot encoded amino acids
        codes = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 
                 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        code_to_int = {c:i for i,c in enumerate(codes)}     
        df['residue'] = df['residue'].map(code_to_int)
        tensor = torch.from_numpy(df['residue'].values)
        onehot = torch.nn.functional.one_hot(tensor).to(torch.float32)
        self.features = torch.cat((self.features, onehot), dim=1)
        

    def __getitem__(self, index):
        features = self.features[index]
        label = self.y[index]
        return features, label

    def __len__(self):
        return self.y.shape[0]
    
    
def get_dataloaders_hotspot_v3_2(batch_size, train_csv_path, test_csv_path, balanced=False, num_workers=0, num_classes=2):

    if num_classes == 2:
        train_dataset = HotspotDataset_v3_2_2class(csv_path=train_csv_path)
        test_dataset = HotspotDataset_v3_2_2class(csv_path=test_csv_path)
    elif num_classes == 3:
        raise NotImplementedError('Not implemented yet')
    else:
        raise ValueError('num_classes option invalid')        
    
    
    if balanced:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=False,
                                  sampler=BalancedBatchSampler(train_dataset, labels=train_dataset.y),
                                  num_workers=num_workers)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=num_workers)

        
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader


#############################################

class HotspotDataset_v4_2class(Dataset):

    def __init__(self, csv_path):
    
        feature_list = ['avg bond number', 'Hbond', 
            'Hphob', 'consurf', "B' side chain", "hotspot ratio"]
        
        df = pd.read_csv(csv_path)
        self.y = torch.from_numpy(df['2-class'].values).to(torch.int64)
        self.features = torch.from_numpy(df[feature_list].values).to(torch.float32)
        
        # convert aa char to int
        codes = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 
                 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        code_to_int = {c:i for i,c in enumerate(codes)}     
        self.residues = df['residue'].map(code_to_int)

    def __getitem__(self, index):
        features = self.features[index]
        residue = self.residues[index]
        label = self.y[index]
        return (features, residue), label

    def __len__(self):
        return self.y.shape[0]


def get_dataloaders_hotspot_v4(batch_size, train_csv_path, test_csv_path, balanced=False, num_workers=0, num_classes=2):

    if num_classes == 2:
        train_dataset = HotspotDataset_v4_2class(csv_path=train_csv_path)
        test_dataset = HotspotDataset_v4_2class(csv_path=test_csv_path)
    elif num_classes == 3:
        raise NotImplementedError('Not implemented yet')
    else:
        raise ValueError('num_classes option invalid')        
    
    
    if balanced:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=False,
                                  sampler=BalancedBatchSampler(train_dataset, labels=train_dataset.y),
                                  num_workers=num_workers)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=num_workers)

        
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader


#############################################

class HotspotDataset_v4_2_2class(Dataset):

    def __init__(self, csv_path):
    
        feature_list = ['avg bond number', 'Hbond', 
            'Hphob', 'consurf', "B' side chain"]
        
        df = pd.read_csv(csv_path)
        self.y = torch.from_numpy(df['2-class'].values).to(torch.int64)
        self.features = torch.from_numpy(df[feature_list].values).to(torch.float32)
        
        # convert aa char to int
        codes = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 
                 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        code_to_int = {c:i for i,c in enumerate(codes)}     
        self.residues = df['residue'].map(code_to_int)

    def __getitem__(self, index):
        features = self.features[index]
        residue = self.residues[index]
        label = self.y[index]
        return (features, residue), label

    def __len__(self):
        return self.y.shape[0]

    
class HotspotDataset_v4_2_3class(Dataset):

    def __init__(self, csv_path):
    
        feature_list = ['avg bond number', 'Hbond', 
            'Hphob', 'consurf', "B' side chain"]
        
        df = pd.read_csv(csv_path)
        self.y = torch.from_numpy(df['3-class'].values).to(torch.int64)
        self.features = torch.from_numpy(df[feature_list].values).to(torch.float32)
        
        # convert aa char to int
        codes = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 
                 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        code_to_int = {c:i for i,c in enumerate(codes)}     
        self.residues = df['residue'].map(code_to_int)

    def __getitem__(self, index):
        features = self.features[index]
        residue = self.residues[index]
        label = self.y[index]
        return (features, residue), label

    def __len__(self):
        return self.y.shape[0]

class Fireman(Dataset):

    def __init__(self, csv_path):
        feature_list = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']
        df = pd.read_csv(csv_path)
        self.y = torch.from_numpy(df['response'].values).to(torch.int64)
        self.features = torch.from_numpy(df[feature_list].values).to(torch.float32)
        
        # convert aa char to int

    def __getitem__(self, index):
        features = self.features[index]
        label = self.y[index]
        return features, label

    def __len__(self):
        return self.y.shape[0]    

def get_data_loaders_fireman(batch_size, train_csv_path, valid_csv_path, test_csv_path, balanced=True, num_workers=0, num_classes=16):
    train_dataset = Fireman(csv_path=train_csv_path)
    valid_dataset = Fireman(csv_path=valid_csv_path)
    test_dataset = Fireman(csv_path=test_csv_path)

    if balanced:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=False,
                                  sampler=BalancedBatchSampler(train_dataset, labels=train_dataset.y),
                                  num_workers=num_workers)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=num_workers)
    
    valid_loader = DataLoader(dataset=valid_dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             num_workers=num_workers)
        
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, valid_loader, test_loader

def get_dataloaders_hotspot_v4_2(batch_size, train_csv_path, test_csv_path, balanced=False, num_workers=0, num_classes=2):

    if num_classes == 2:
        train_dataset = HotspotDataset_v4_2_2class(csv_path=train_csv_path)
        test_dataset = HotspotDataset_v4_2_2class(csv_path=test_csv_path)
    elif num_classes == 3:
        train_dataset = HotspotDataset_v4_2_3class(csv_path=train_csv_path)
        test_dataset = HotspotDataset_v4_2_3class(csv_path=test_csv_path)
    else:
        raise ValueError('num_classes option invalid')        
    
    
    if balanced:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=False,
                                  sampler=BalancedBatchSampler(train_dataset, labels=train_dataset.y),
                                  num_workers=num_workers)

    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=num_workers)

        
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader


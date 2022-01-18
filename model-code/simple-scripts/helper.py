import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision


def parse_cmdline_args(parser):

    parser.add_argument(
        '--cuda',
        type=int,
        default=-1,
        help='Which GPU device to use. Uses cpu if `-1`.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        help='Which random seed to use. No random seed if `-1`.'
    )

    parser.add_argument(
        '--numworkers',
        type=int,
        default=0,
        help='How many workers to use for the data loader.'
    )

    parser.add_argument(
        '--learningrate',
        type=float,
        default=0.0005,
        help='Learning rate.'
    )

    parser.add_argument(
        '--batchsize',
        type=int,
        default=32,
        help='Batch size.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs.'
    )

    parser.add_argument(
        '--loss_print_interval',
        type=int,
        default=50,
        help='How frequently to print the loss during training.'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='output_dir',
        help='Location for the training log and best'
             ' model (based on validation set MAE).'
    )

    parser.add_argument(
        '--overwrite',
        type=str,
        choices=['true', 'false'],
        default='false',
        help='Whether to overwrite the results folder.'
        )

    parser.add_argument(
        '--logfile_path',
        type=str,
        default='training.log',
        help='Location for the log file.'
        )

    parser.set_defaults(feature=True)
    args = parser.parse_args()

    d = {'true': True,
         'false': False}

    args.overwrite = d[args.overwrite]

    return args

############################################################
# Metrics
############################################################


def compute_mae_and_rmse(model, data_loader, device, label_from_logits_func):
    with torch.no_grad():

        mae, mse, num_examples = 0., 0., 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            predicted_labels = label_from_logits_func(logits)

            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(predicted_labels - targets))
            mse += torch.sum((predicted_labels - targets)**2)

        mae = mae / num_examples
        mse = mse / num_examples
        return mae, torch.sqrt(mse)


############################################################
# ResNet-34
############################################################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):

    def __init__(self, block, layers, num_classes, grayscale,
                 resnet34_avg_poolsize=4):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AvgPool2d(resnet34_avg_poolsize)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)


def resnet34base(num_classes, grayscale, resnet34_avg_poolsize=4):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale,
                   resnet34_avg_poolsize=resnet34_avg_poolsize)
    return model


############################################################
# Dataset
############################################################

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

    def _get_label(self, dataset, idx, labels=None):
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


class FiremanDataset(Dataset):

    def __init__(self, csv_path):

        feature_list = ['V1', 'V2', 'V3', 'V4', 'V5',
                        'V6', 'V7', 'V8', 'V9', 'V10']

        df = pd.read_csv(csv_path)
        self.y = torch.from_numpy(df['response'].values).to(torch.int64)
        self.features = torch.from_numpy(
            df[feature_list].values).to(torch.float32)

    def __getitem__(self, index):
        features = self.features[index]
        label = self.y[index]
        return features, label

    def __len__(self):
        return self.y.shape[0]


def get_dataloaders_fireman(batch_size, train_csv_path,
                            valid_csv_path, test_csv_path,
                            balanced=False, num_workers=0, num_classes=16):

    train_dataset = FiremanDataset(csv_path=train_csv_path)
    valid_dataset = FiremanDataset(csv_path=valid_csv_path)
    test_dataset = FiremanDataset(csv_path=test_csv_path)

    if balanced:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=False,
                                  sampler=BalancedBatchSampler(
                                      train_dataset, labels=train_dataset.y),
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

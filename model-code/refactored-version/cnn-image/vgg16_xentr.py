# coding: utf-8

# Imports

import os
import json
import pandas as pd
import time
import torch
import torch.nn as nn
import types
import argparse
import sys
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler


# ### from local .py files

from helper_files.trainingeval import (
    iteration_logging,
    epoch_logging,
    aftertraining_logging,
    save_predictions,
    create_logfile,
)
from helper_files.trainingeval import compute_per_class_mae, compute_selfentropy_for_mae
from helper_files.helper import set_all_seeds, set_deterministic
from helper_files.plotting import plot_training_loss, plot_mae, plot_accuracy
from helper_files.plotting import plot_per_class_mae
from helper_files.dataset import get_labels_from_loader
from helper_files.parser import parse_cmdline_args


# Argparse helper
parser = argparse.ArgumentParser()
args = parse_cmdline_args(parser)

##########################
# Settings and Setup
##########################

NUM_WORKERS = args.numworkers
LEARNING_RATE = args.learningrate
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
SKIP_TRAIN_EVAL = args.skip_train_eval
SAVE_MODELS = args.save_models

if args.cuda >= 0 and torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{args.cuda}")
else:
    DEVICE = torch.device("cpu")

if args.seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = args.seed

PATH = args.outpath
if not os.path.exists(PATH):
    os.mkdir(PATH)

cuda_device = DEVICE
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
else:
    cuda_version = "NA"

info_dict = {
    "settings": {
        "script": os.path.basename(__file__),
        "pytorch version": torch.__version__,
        "cuda device": str(cuda_device),
        "cuda version": cuda_version,
        "random seed": RANDOM_SEED,
        "learning rate": LEARNING_RATE,
        "num epochs": NUM_EPOCHS,
        "batch size": BATCH_SIZE,
        "output path": PATH,
        "training logfile": os.path.join(PATH, "training.log"),
    }
}

create_logfile(info_dict)

# Deterministic CUDA & cuDNN behavior and random seeds
# set_deterministic()
set_all_seeds(RANDOM_SEED)


###################
# Dataset
###################

if args.dataset == "mnist":
    from helper_files.constants import MNIST_INFO as DATASET_INFO
    from torchvision.datasets import MNIST as PyTorchDataset

elif args.dataset == "morph2":
    from helper_files.constants import MORPH2_INFO as DATASET_INFO
    from helper_files.dataset import Morph2Dataset as PyTorchDataset

elif args.dataset == "morph2-balanced":
    from helper_files.constants import MORPH2_BALANCED_INFO as DATASET_INFO
    from helper_files.dataset import Morph2Dataset as PyTorchDataset

elif args.dataset == 'afad-balanced':
    from helper_files.constants import AFAD_BALANCED_INFO as DATASET_INFO
    from helper_files.dataset import AFADDataset as PyTorchDataset

elif args.dataset == 'aes':
    from helper_files.constants import AES_INFO as DATASET_INFO
    from helper_files.dataset import AesDataset as PyTorchDataset

elif args.dataset == 'aes-nature':
    from helper_files.constants import AES_INFO as DATASET_INFO
    from helper_files.dataset import AesDatasetNature as PyTorchDataset

elif args.dataset == 'aes-people':
    from helper_files.constants import AES_INFO as DATASET_INFO
    from helper_files.dataset import AesDatasetPeople as PyTorchDataset

elif args.dataset == 'aes-urban':
    from helper_files.constants import AES_INFO as DATASET_INFO
    from helper_files.dataset import AesDatasetUrban as PyTorchDataset

elif args.dataset == 'aes-animals':
    from helper_files.constants import AES_INFO as DATASET_INFO
    from helper_files.dataset import AesDatasetAnimal as PyTorchDataset

else:
    raise ValueError("Dataset choice not supported")

###################
# Transforms
###################

train_transform_list = [
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

valid_transform_list = [
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

if args.dataset == "mnist":
    train_transform_list.insert(0, transforms.Grayscale(num_output_channels=3))
    valid_transform_list.insert(0, transforms.Grayscale(num_output_channels=3))

train_transform = transforms.Compose(train_transform_list)
validation_transform = transforms.Compose(valid_transform_list)

###################
# Dataset
###################

if args.dataset == "mnist":

    NUM_CLASSES = 10

    train_dataset = PyTorchDataset(
        root="./datasets", train=True, download=True, transform=train_transform
    )

    valid_dataset = PyTorchDataset(
        root="./datasets", train=True, transform=validation_transform, download=False
    )

    test_dataset = PyTorchDataset(
        root="./datasets", train=False, transform=validation_transform, download=False
    )

    train_indices = torch.arange(1000, 60000)
    valid_indices = torch.arange(0, 1000)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # SubsetRandomSampler shuffles
        drop_last=True,
        num_workers=NUM_WORKERS,
        sampler=train_sampler,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        sampler=valid_sampler,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

else:
    
    if args.dataset_train_csv_path:
        DATASET_INFO['TRAIN_CSV_PATH'] = args.dataset_train_csv_path

    if args.dataset_valid_csv_path:
        DATASET_INFO['VALID_CSV_PATH'] = args.dataset_valid_csv_path
        
    if args.dataset_test_csv_path:
        DATASET_INFO['TEST_CSV_PATH'] = args.dataset_test_csv_path
        
    if args.dataset_img_path:
        DATASET_INFO['IMAGE_PATH'] = args.dataset_img_path

    df = pd.read_csv(DATASET_INFO["TRAIN_CSV_PATH"], index_col=0)
    classes = df[DATASET_INFO["CLASS_COLUMN"]].values
    del df
    train_labels = torch.tensor(classes, dtype=torch.float)
    NUM_CLASSES = torch.unique(train_labels).size()[0]
    del classes

    train_dataset = PyTorchDataset(
        csv_path=DATASET_INFO["TRAIN_CSV_PATH"],
        img_dir=DATASET_INFO["IMAGE_PATH"],
        transform=train_transform,
    )

    test_dataset = PyTorchDataset(
        csv_path=DATASET_INFO["TEST_CSV_PATH"],
        img_dir=DATASET_INFO["IMAGE_PATH"],
        transform=validation_transform,
    )

    valid_dataset = PyTorchDataset(
        csv_path=DATASET_INFO["VALID_CSV_PATH"],
        img_dir=DATASET_INFO["IMAGE_PATH"],
        transform=validation_transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

info_dict["dataset"] = DATASET_INFO
info_dict["settings"]["num classes"] = NUM_CLASSES


##########################
# MODEL
##########################

model = torch.hub.load("pytorch/vision:v0.10.0", "vgg16_bn", pretrained=True)
model.classifier[-1] = torch.nn.Linear(4096, NUM_CLASSES)


def forward_with_probas(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, start_dim=1)
    logits = self.classifier(x)
    probas = torch.nn.functional.softmax(logits, dim=1)
    return logits, probas


model.forward = types.MethodType(forward_with_probas, model)

model.to(DEVICE)
###########################################
# Initialize Loss and Optimizer
###########################################


if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
else:
    raise ValueError('--optimizer must be "adam" or "sgd"')

if args.scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", verbose=True
    )


start_time = time.time()

best_mae, best_rmse, best_epoch = 999, 999, -1

info_dict["training"] = {
    "num epochs": NUM_EPOCHS,
    "iter per epoch": len(train_loader),
    "minibatch loss": [],
    "epoch train mae": [],
    "epoch train rmse": [],
    "epoch train acc": [],
    "epoch valid mae": [],
    "epoch valid rmse": [],
    "epoch valid acc": [],
    "best running mae": np.infty,
    "best running rmse": np.infty,
    "best running acc": 0.0,
    "best running epoch": -1,
}

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # FORWARD AND BACK PROP
        logits, probas = model(features)

        loss = torch.nn.functional.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ITERATION LOGGING
        iteration_logging(
            info_dict=info_dict,
            batch_idx=batch_idx,
            loss=loss,
            train_dataset=train_dataset,
            frequency=50,
            epoch=epoch,
        )

    # EPOCH LOGGING
    # function saves best model as best_model.pt
    best_mae = epoch_logging(
        info_dict=info_dict,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        which_model="categorical",
        loss=loss,
        epoch=epoch,
        start_time=start_time,
        skip_train_eval=SKIP_TRAIN_EVAL,
    )

    if args.scheduler:
        scheduler.step(info_dict["training"]["epoch valid rmse"][-1])

# ####### AFTER TRAINING EVALUATION
# function saves last model as last_model.pt
info_dict["last"] = {}
aftertraining_logging(
    model=model,
    which="last",
    info_dict=info_dict,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    which_model="categorical",
    start_time=start_time,
)

info_dict["best"] = {}
aftertraining_logging(
    model=model,
    which="best",
    info_dict=info_dict,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    which_model="categorical",
    start_time=start_time,
)

# ######### MAKE PLOTS ######
plot_training_loss(info_dict=info_dict, averaging_iterations=100)
plot_mae(info_dict=info_dict)
plot_accuracy(info_dict=info_dict)

# ######### PER-CLASS MAE PLOT #######

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS,
)

for best_or_last in ("best", "last"):

    model.load_state_dict(
        torch.load(
            os.path.join(
                info_dict["settings"]["output path"], f"{best_or_last}_model.pt"
            )
        )
    )

    names = {0: "train", 1: "test"}
    for i, data_loader in enumerate([train_loader, test_loader]):

        true_labels = get_labels_from_loader(data_loader)

        # ######### SAVE PREDICTIONS ######
        all_probas, all_predictions = save_predictions(
            model=model,
            which=best_or_last,
            which_model="categorical",
            info_dict=info_dict,
            data_loader=data_loader,
            prefix=names[i],
        )

        errors, counts = compute_per_class_mae(
            actual=true_labels.numpy(), predicted=all_predictions.numpy()
        )

        info_dict[f"per-class mae {names[i]} ({best_or_last} model)"] = errors

        # actual_selfentropy_best, best_selfentropy_best =\
        #    compute_selfentropy_for_mae(errors_best)

        # info_dict['test set mae self-entropy'] = actual_selfentropy_best.item()
        # info_dict['ideal test set mae self-entropy'] = best_selfentropy_best.item()

plot_per_class_mae(info_dict)

# ######## CLEAN UP ########
json.dump(info_dict, open(os.path.join(PATH, "info_dict.json"), "w"), indent=4)

if not SAVE_MODELS:
    os.remove(os.path.join(PATH, "best_model.pt"))
    os.remove(os.path.join(PATH, "last_model.pt"))

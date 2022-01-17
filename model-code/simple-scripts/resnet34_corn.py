import argparse
from functools import partial
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision import transforms

# Import from local helper file
from helper import parse_cmdline_args
from helper import compute_mae_and_rmse
from helper import resnet34base


# Argparse helper
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parse_cmdline_args(parser)

##########################
# Settings and Setup
##########################

NUM_WORKERS = args.numworkers
LEARNING_RATE = args.learningrate
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
OUTPUT_DIR = args.output_dir
LOSS_PRINT_INTERVAL = args.loss_print_interval

if os.path.exists(args.output_dir):
    raise ValueError('Output directory already exists.')
else:
    os.makedirs(args.output_dir)
BEST_MODEL_PATH = os.path.join(args.output_dir, 'best_model.pt')
LOGFILE_PATH = os.path.join(args.output_dir, 'training.log')

if args.cuda >= 0 and torch.cuda.is_available():
    DEVICE = torch.device(f'cuda:{args.cuda}')
else:
    DEVICE = torch.device('cpu')

if args.seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = args.seed


############################
# Dataset
############################

def train_transform():
    return transforms.Compose([transforms.ToTensor()])


def validation_transform():
    return transforms.Compose([transforms.ToTensor()])


NUM_CLASSES = 10
GRAYSCALE = True
RESNET34_AVGPOOLSIZE = 1

train_dataset = MNIST(root='./datasets',
                      train=True,
                      download=True,
                      transform=train_transform())

valid_dataset = MNIST(root='./datasets',
                      train=True,
                      transform=validation_transform(),
                      download=False)

test_dataset = MNIST(root='./datasets',
                     train=False,
                     transform=validation_transform(),
                     download=False)

train_indices = torch.arange(1000, 60000)
valid_indices = torch.arange(0, 1000)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,  # SubsetRandomSampler shuffles
                          drop_last=True,
                          num_workers=NUM_WORKERS,
                          sampler=train_sampler)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=NUM_WORKERS,
                          sampler=valid_sampler)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)


##########################
# MODEL
##########################

model = resnet34base(
    num_classes=NUM_CLASSES,
    grayscale=GRAYSCALE,
    resnet34_avg_poolsize=RESNET34_AVGPOOLSIZE)


model.output_layer = torch.nn.Linear(512, NUM_CLASSES-1)


def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)

    x = x.view(x.size(0), -1)
    logits = self.output_layer(x)
    return logits


def add_method(obj, func):
    'Bind a function and store it in an object'
    setattr(obj, func.__name__, partial(func, obj))


add_method(model, forward)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


#######################################
# Loss and Evaluation Functions
#######################################


def loss_corn(logits, y_train, num_classes):
    sets = []
    for i in range(num_classes-1):
        label_mask = y_train > i-1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(F.logsigmoid(pred)*train_labels
                          + (F.logsigmoid(pred) - pred)*(1-train_labels)
                          )
        losses += loss
    return losses/num_examples


def label_from_logits(logits):
    """ Converts logits to class labels.
    This is function is specific to CORN.
    """
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels


#######################################
# Training
#######################################


best_valid_mae = torch.tensor(float('inf'))

s = (f'Script: {__file__}\n'
     f'PyTorch version: {torch.__version__}\n'
     f'Device: {DEVICE}\n'
     f'Learning rate: {LEARNING_RATE}\n'
     f'Batch size: {BATCH_SIZE}\n')

print(s)
with open(LOGFILE_PATH, 'w') as f:
    f.write(f'{s}\n')

start_time = time.time()

for epoch in range(1, NUM_EPOCHS+1):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # FORWARD AND BACK PROP
        logits = model(features)

        # CORN loss
        loss = loss_corn(logits, targets, NUM_CLASSES)
        # ##--------------------------------------------------------------------###

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if not batch_idx % LOSS_PRINT_INTERVAL:
            s = (f'Epoch: {epoch:03d}/{NUM_EPOCHS:03d} | '
                 f'Batch {batch_idx:04d}/'
                 f'{len(train_dataset)//BATCH_SIZE:04d} | '
                 f'Loss: {loss:.4f}')
            print(s)
            with open(LOGFILE_PATH, 'a') as f:
                f.write(f'{s}\n')

    # Logging: Evaluate after epoch
    model.eval()
    with torch.no_grad():
        valid_mae, valid_rmse = compute_mae_and_rmse(
            model=model,
            data_loader=valid_loader,
            device=DEVICE,
            label_from_logits_func=label_from_logits
        )

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        s = (f'MAE Current Valid: {valid_mae:.2f} Ep. {epoch}'
             f' | Best Valid: {best_valid_mae:.2f} Ep. {best_epoch}')
        s += f'\nTime elapsed: {(time.time() - start_time)/60:.2f} min'
        print(s)
        with open(LOGFILE_PATH, 'a') as f:
            f.write('%s\n' % s)


# Final
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()
with torch.no_grad():

    train_mae, train_rmse = compute_mae_and_rmse(
        model=model,
        data_loader=train_loader,
        device=DEVICE,
        label_from_logits_func=label_from_logits
        )

    valid_mae, valid_rmse = compute_mae_and_rmse(
        model=model,
        data_loader=valid_loader,
        device=DEVICE,
        label_from_logits_func=label_from_logits
        )

    test_mae, test_rmse = compute_mae_and_rmse(
        model=model,
        data_loader=valid_loader,
        device=DEVICE,
        label_from_logits_func=label_from_logits
        )

s = ('\n\n=========================================\n\n'
     'Performance of best model based on validation set MAE:'
     f'Train MAE / RMSE: {train_mae:.2f} / {train_rmse:.2f}'
     f'Valid MAE / RMSE: {valid_mae:.2f} / {valid_rmse:.2f}'
     f'Test  MAE / RMSE: {test_mae:.2f} / {test_rmse:.2f}')

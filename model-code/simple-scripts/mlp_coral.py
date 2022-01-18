import argparse
import os
import shutil
import time

import torch
import torch.nn.functional as F

# Import from local helper file
from helper import parse_cmdline_args
from helper import compute_mae_and_rmse
from helper import get_dataloaders_fireman


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
    if args.overwrite:
        shutil.rmtree(args.output_dir)
    else:
        raise ValueError('Output directory already exists.')

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


NUM_CLASSES = 16
NUM_FEATURES = 10

train_loader, valid_loader, test_loader = get_dataloaders_fireman(
    batch_size=BATCH_SIZE,
    train_csv_path='./datasets/fireman_example_balanced_train.csv',
    valid_csv_path='./datasets/fireman_example_balanced_valid.csv',
    test_csv_path='./datasets/fireman_example_balanced_test.csv',
    balanced=True,
    num_workers=NUM_WORKERS,
    num_classes=NUM_CLASSES)


##########################
# MODEL
##########################

class MultilayerPerceptron(torch.nn.Module):

    def __init__(self, num_features, num_classes,
                 num_hidden_1, num_hidden_2):
        super().__init__()

        self.num_classes = num_classes
        self.my_network = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_features, num_hidden_1, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.BatchNorm1d(num_hidden_1),

            # 2nd hidden layer
            torch.nn.Linear(num_hidden_1, num_hidden_2, bias=False),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.BatchNorm1d(num_hidden_2),

            torch.nn.Linear(num_hidden_2, 1, bias=False)
        )
        self.output_biases = torch.nn.Parameter(
            torch.zeros(NUM_CLASSES-1).float())

    def forward(self, x):
        logits = self.my_network(x)
        logits = logits + self.output_biases.view(1, -1)
        return logits


if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)
model = MultilayerPerceptron(num_features=NUM_FEATURES,
                             num_hidden_1=300,
                             num_hidden_2=300,
                             num_classes=16)

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


#######################################
# Utility Functions
#######################################

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


def loss_coral(logits, levels):
    val = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels)),
           dim=1))
    return torch.mean(val)


def label_from_logits(logits):
    """ Converts logits to class labels.
    This is function is specific to CORAL.
    """
    probas = torch.sigmoid(logits)
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

        # CORAL loss
        levels = levels_from_labelbatch(
            targets,
            num_classes=NUM_CLASSES).type_as(logits)
        loss = loss_coral(logits, levels)
        # ##--------------------------------------------------------------------###

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if not batch_idx % LOSS_PRINT_INTERVAL:
            s = (f'Epoch: {epoch:03d}/{NUM_EPOCHS:03d} | '
                 f'Batch {batch_idx:04d}/'
                 f'{len(train_loader):04d} | '
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

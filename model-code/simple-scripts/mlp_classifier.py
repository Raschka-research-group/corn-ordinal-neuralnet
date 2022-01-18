import argparse
import os
import shutil
import time

import torch

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

            torch.nn.Linear(num_hidden_2, self.num_classes)
        )

    def forward(self, x):
        logits = self.my_network(x)
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


def label_from_logits(logits):
    _, predicted_labels = torch.max(logits, 1)
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

        # Cross Entropy loss
        loss = torch.nn.functional.cross_entropy(logits, targets)
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

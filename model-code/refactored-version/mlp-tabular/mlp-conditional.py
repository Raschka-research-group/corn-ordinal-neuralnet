#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'watermark')
# get_ipython().run_line_magic('watermark', "-a 'Sebastian Raschka' -v -p torch")


# # MLP

# ## Imports

# In[2]:

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

# In[3]:


import sys
sys.path.insert(0, "./helper_files") # to include ../helper_evaluate.py etc.

# From local helper files
from helper_utils import set_all_seeds, set_deterministic
from helper_evaluate import compute_confusion_matrix, compute_accuracy, get_labels_and_predictions
from helper_evaluate import compute_mae_and_mse
from helper_train import train_model_v2, create_logfile

from helper_plotting import plot_training_loss, plot_accuracy, plot_confusion_matrix
from helper_data import get_data_loaders_fireman
from helper_parser import parse_cmdline_args


# Other libraries
# from mlxtend.evaluate import confusion_matrix 
# from mlxtend.plotting import plot_confusion_matrix
# from mlxtend.evaluate import scoring
import matplotlib.pyplot as plt


# ## Settings and Dataset

# In[4]:


##########################
### SETTINGS
##########################
parser = argparse.ArgumentParser()
args = parse_cmdline_args(parser)


RANDOM_SEED = args.seed
BATCH_SIZE = args.batchsize
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.learningrate
NUM_FEATURES = 10

PATH = args.outpath
if not os.path.exists(PATH):
  os.mkdir(PATH)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[5]:


set_all_seeds(RANDOM_SEED)
set_deterministic()


# In[6]:


train_loader, valid_loader, test_loader = get_data_loaders_fireman(
    batch_size=BATCH_SIZE,
    train_csv_path='../../../datasets/firemen/fireman_example_balanced_train.csv',
    valid_csv_path='../../../datasets/firemen/fireman_example_balanced_valid.csv',
    test_csv_path='../../../datasets/firemen/fireman_example/fireman_example_balanced_test.csv',
    balanced=True,
    num_classes=16,
)

info_dict = {
  'settings': {
      'script': os.path.basename(__file__),
      'pytorch version': torch.__version__,
      'random seed': RANDOM_SEED,
      'learning rate': LEARNING_RATE,
      'num epochs': NUM_EPOCHS,
      'batch size': BATCH_SIZE,
      'output path': PATH,
      'training logfile': os.path.join(PATH, 'training.log')}
}

create_logfile(info_dict)

# ## Model

# In[7]:


##########################
### MODEL
##########################


class MultilayerPerceptron(torch.nn.Module):

    def __init__(self, num_features, num_classes, 
                 num_hidden_1,num_hidden_2):
        super().__init__()

        self.num_classes = num_classes

        # self.embedding = torch.nn.Embedding(
        #     num_embeddings=20, embedding_dim=embedding_dim)

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

            torch.nn.Linear(num_hidden_2, (self.num_classes-1))
        )

    def forward(self, x):
        logits = self.my_network(x)
        probas = torch.sigmoid(logits)
        probas = torch.cumprod(probas, dim=1)
        return logits, probas


torch.manual_seed(RANDOM_SEED)
model = MultilayerPerceptron(num_features=NUM_FEATURES,
                             num_hidden_1=300,
                             num_hidden_2=300,
                             num_classes=16)
                             
model = model.to(DEVICE)


# In[8]:


#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.5)

#optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.0, weight_decay=0.01)

#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                       factor=0.5,
#                                                       mode='min',
#                                                       verbose=True)


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.2)


# In[9]:


train_model_v2(
    model=model,
    num_epochs=NUM_EPOCHS,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    #scheduler=scheduler,
    #scheduler_on='minibatch_loss',
    verbose=1,
    device=DEVICE,
    logging_interval=100,
    info_dict=info_dict,
    with_embedding=False,
    which_model='conditional')


json.dump(info_dict, open(os.path.join(PATH, 'info_dict.json'), 'w'), indent=4)

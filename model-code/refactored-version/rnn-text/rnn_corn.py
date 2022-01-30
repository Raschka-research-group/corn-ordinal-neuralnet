# coding: utf-8

# Like v2, and in contrast to v1, this version removes the cumprod from the forward pass

# In addition, it uses a different conditional loss function compared to v2.
# Here, the loss is computed as the average loss of the total samples, 
# instead of firstly averaging the cross entropy inside each task and then averaging over tasks equally. 
# The weight of each task will be adjusted
# for the sample size used for training each task naturally without manually setting the weights.

# Imports


import os
import json
import pandas as pd
import time
import torch
import torch.nn as nn
import argparse
import sys
import numpy as np
import torchtext
import random

from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler


# ### from local .py files
if __name__ == '__main__':
    sys.path.insert(0, "./helper_files")  # to include ../layer.py etc.
    from trainingeval import (iteration_logging, epoch_logging,
                              aftertraining_logging, save_predictions,
                              create_logfile)
    from trainingeval import compute_per_class_mae, compute_selfentropy_for_mae
    from resnet34 import BasicBlock
    from dataset import levels_from_labelbatch
    from losses import loss_conditional_v2
    from helper import set_all_seeds, set_deterministic
    from plotting import plot_training_loss, plot_mae, plot_accuracy
    from plotting import plot_per_class_mae
    from dataset import get_labels_from_loader
    from parser import parse_cmdline_args


    # Argparse helper
    parser = argparse.ArgumentParser()
    args = parse_cmdline_args(parser)

    ##########################
    # Settings and Setup
    ##########################


    NUM_WORKERS = args.numworkers
    LEARNING_RATE = args.learningrate
    VOCABULARY_SIZE = args.vocabulary_size
    EMBEDDING_DIM = args.embedding_dim
    HIDDEN_DIM = args.hidden_dim
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    SKIP_TRAIN_EVAL = args.skip_train_eval
    SAVE_MODELS = args.save_models

    if args.cuda >= 0 and torch.cuda.is_available():
        DEVICE = torch.device(f'cuda:{args.cuda}')
    else:
        DEVICE = torch.device('cpu')

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
        cuda_version = 'NA'

    info_dict = {
        'settings': {
            'script': os.path.basename(__file__),
            'pytorch version': torch.__version__,
            'cuda device': str(cuda_device),
            'cuda version': cuda_version,
            'random seed': RANDOM_SEED,
            'learning rate': LEARNING_RATE,
            'num epochs': NUM_EPOCHS,
            'batch size': BATCH_SIZE,
            'vocabulary size': VOCABULARY_SIZE,
            'embedding dim': EMBEDDING_DIM,
            'hidden dim': HIDDEN_DIM,
            'output path': PATH,
            'training logfile': os.path.join(PATH, 'training.log')}
    }

    create_logfile(info_dict)

    # Deterministic CUDA & cuDNN behavior and random seeds
    #set_deterministic()
    set_all_seeds(RANDOM_SEED)


    ###################
    # Dataset
    ###################
    if args.dataset == 'tripadvisor':
        from constants import TRIPADVISOR_BALANCED_INFO as DATASET_INFO
    elif args.dataset == 'coursera':
        from constants import COURSERA_BALANCED_INFO as DATASET_INFO
    else:
        raise ValueError('Dataset choice not supported')

    #number of classes fixed
    NUM_CLASSES = 5

    TEXT = torchtext.legacy.data.Field(
        tokenize='spacy', # default splits on whitespace
        tokenizer_language='en_core_web_sm',
        include_lengths=True # NEW
    )

    LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)

    fields = [('TEXT_COLUMN_NAME', TEXT), ('LABEL_COLUMN_NAME', LABEL)]

    dataset = torchtext.legacy.data.TabularDataset(
        path=DATASET_INFO['DATA_PATH'], format='csv',
        skip_header=True, fields=fields)


    train_data, test_data = dataset.split(
        split_ratio=[0.8, 0.2],
        random_state=random.seed(RANDOM_SEED))


    train_data, valid_data = train_data.split(
        split_ratio=[0.85, 0.15],
        random_state=random.seed(RANDOM_SEED))


    TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE)
    LABEL.build_vocab(train_data)


    train_loader, valid_loader, test_loader =     torchtext.legacy.data.BucketIterator.splits(
            (train_data, valid_data, test_data), 
            batch_size=BATCH_SIZE,
            sort_within_batch=True, # NEW. necessary for packed_padded_sequence
                 sort_key=lambda x: len(x.TEXT_COLUMN_NAME),
            device=DEVICE
    )
    info_dict['dataset'] = DATASET_INFO
    info_dict['settings']['num classes'] = NUM_CLASSES


    ##########################
    # MODEL
    ##########################


    class RNN(torch.nn.Module):
    
        def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,num_classes):
            super().__init__()

            self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
            #self.rnn = torch.nn.RNN(embedding_dim,
            #                        hidden_dim,
            #                        nonlinearity='relu')
            self.rnn = torch.nn.LSTM(embedding_dim,
                                     hidden_dim)        
            
            self.fc = torch.nn.Linear(hidden_dim, num_classes-1)
            self.num_classes = num_classes

        def forward(self, text, text_length):
            # text dim: [sentence length, batch size]
            
            embedded = self.embedding(text)
            # ebedded dim: [sentence length, batch size, embedding dim]
            
            ## NEW
            packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))
            
            packed_output, (hidden, cell) = self.rnn(embedded)
            # output dim: [sentence length, batch size, hidden dim]
            # hidden dim: [1, batch size, hidden dim]

            hidden.squeeze_(0)
            # hidden dim: [batch size, hidden dim]
            
            output = self.fc(hidden)
            #add corn layer
            logits = output.view(-1, (self.num_classes-1))
            probas = torch.sigmoid(logits)
            return logits, probas

    torch.manual_seed(RANDOM_SEED)
    model = RNN(input_dim=len(TEXT.vocab),
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=NUM_CLASSES, # could use 1 for binary classification
                num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)


    ###########################################
    # Initialize Cost, Model, and Optimizer
    ###########################################

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                    momentum=0.9)
    else:
        raise ValueError('--optimizer must be "adam" or "sgd"')

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               verbose=True)

    start_time = time.time()

    best_mae, best_rmse, best_epoch = 999, 999, -1

    info_dict['training'] = {
             'num epochs': NUM_EPOCHS,
             'iter per epoch': len(train_loader),
             'minibatch loss': [],
             'epoch train mae': [],
             'epoch train rmse': [],
             'epoch train acc': [],
             'epoch valid mae': [],
             'epoch valid rmse': [],
             'epoch valid acc': [],
             'best running mae': np.infty,
             'best running rmse': np.infty,
             'best running acc': 0.,
             'best running epoch': -1
    }

    for epoch in range(1, NUM_EPOCHS+1):

        model.train()
        for batch_idx, batch_data in enumerate(train_loader):

            features, text_length = batch_data.TEXT_COLUMN_NAME
            labels = batch_data.LABEL_COLUMN_NAME.to(DEVICE)
            # FORWARD AND BACK PROP
            logits, probas = model(features,text_length)

            # ### Ordinal loss
            loss = loss_conditional_v2(logits, labels, NUM_CLASSES)
            # ##--------------------------------------------------------------------###

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ITERATION LOGGING
            iteration_logging(info_dict=info_dict, batch_idx=batch_idx,
                              loss=loss, train_dataset=train_data,
                              frequency=50, epoch=epoch)

        # EPOCH LOGGING
        # function saves best model as best_model.pt
        best_mae = epoch_logging(info_dict=info_dict,
                                 model=model, train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 which_model='conditional',
                                 loss=loss, epoch=epoch, start_time=start_time,
                                 skip_train_eval=SKIP_TRAIN_EVAL)

        if args.scheduler:
            scheduler.step(info_dict['training']['epoch valid rmse'][-1])


    # ####### AFTER TRAINING EVALUATION
    # function saves last model as last_model.pt
    info_dict['last'] = {}
    aftertraining_logging(model=model, which='last', info_dict=info_dict,
                          train_loader=train_loader,
                          valid_loader=valid_loader, test_loader=test_loader,
                          which_model='conditional',
                          start_time=start_time)

    info_dict['best'] = {}
    aftertraining_logging(model=model, which='best', info_dict=info_dict,
                          train_loader=train_loader,
                          valid_loader=valid_loader, test_loader=test_loader,
                          which_model='conditional',
                          start_time=start_time)

    # ######### MAKE PLOTS ######
    plot_training_loss(info_dict=info_dict, averaging_iterations=100)
    plot_mae(info_dict=info_dict)
    plot_accuracy(info_dict=info_dict)

    # ######### PER-CLASS MAE PLOT #######

    # train_loader = DataLoader(dataset=train_dataset,
    #                           batch_size=512,
    #                           shuffle=False,
    #                           drop_last=False,
    #                           num_workers=NUM_WORKERS)

    for best_or_last in ('best', 'last'):

        model.load_state_dict(torch.load(
            os.path.join(info_dict['settings']['output path'], f'{best_or_last}_model.pt')))

        names = {0: 'train',
                 1: 'test'}
        for i, data_loader in enumerate([train_loader, test_loader]):

            true_labels = get_labels_from_loader(data_loader).to(torch.device('cpu'))

            # ######### SAVE PREDICTIONS ######
            all_probas, all_predictions = save_predictions(model=model,
                                                           which=best_or_last,
                                                           which_model='conditional',
                                                           info_dict=info_dict,
                                                           data_loader=data_loader,
                                                           prefix=names[i])

            errors, counts = compute_per_class_mae(actual=true_labels.numpy(),
                                                   predicted=all_predictions.numpy())

            info_dict[f'per-class mae {names[i]} ({best_or_last} model)'] = errors

            #actual_selfentropy_best, best_selfentropy_best =\
            #    compute_selfentropy_for_mae(errors_best)

            #info_dict['test set mae self-entropy'] = actual_selfentropy_best.item()
            #info_dict['ideal test set mae self-entropy'] = best_selfentropy_best.item()

    plot_per_class_mae(info_dict)

    # ######## CLEAN UP ########
    json.dump(info_dict, open(os.path.join(PATH, 'info_dict.json'), 'w'), indent=4)

    if not SAVE_MODELS:
        os.remove(os.path.join(PATH, 'best_model.pt'))
        os.remove(os.path.join(PATH, 'last_model.pt'))

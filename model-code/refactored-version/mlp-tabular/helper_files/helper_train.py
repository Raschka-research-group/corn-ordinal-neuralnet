from helper_evaluate import compute_accuracy
from helper_evaluate import compute_epoch_loss
from helper_losses import niu_loss, coral_loss
from helper_data import levels_from_labelbatch

import time
import torch
import torch.nn.functional as F

from collections import OrderedDict
import json
import subprocess
import sys

    
def train_classifier_v1(model, num_epochs, train_loader,
                        valid_loader, optimizer,
                        device, logging_interval=50,
                        verbose=1,
                        scheduler=None,
                        scheduler_on='valid_acc',
                        with_embedding=False):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            if with_embedding:
                features, residues = features
                features = features.to(device)
                residues = residues.to(device)
                targets = targets.to(device)
                logits = model(features, residues)
                
            else:
                features = features.to(device)
                targets = targets.to(device)   
                logits = model(features)


            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            minibatch_loss_list.append(loss.item())
            
            if verbose:
                if not batch_idx % logging_interval:
                    print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                          f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                          f'| Loss: {loss:.4f}')
 
        model.eval()
        with torch.no_grad():  # save memory during inference
            train_acc = compute_accuracy(
                model, train_loader, device=device,
                with_embedding=with_embedding)
            valid_acc = compute_accuracy(
                model, valid_loader, device=device,
                with_embedding=with_embedding)
            
            if verbose:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Train: {train_acc :.2f}% '
                      f'| Validation: {valid_acc :.2f}%')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        elapsed = (time.time() - start_time)/60
        if verbose:
            print(f'Time elapsed: {elapsed:.2f} min')
        
        if scheduler is not None:

            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(minibatch_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')
        

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    return minibatch_loss_list, train_acc_list, valid_acc_list



def train_model_v2(model, num_epochs, train_loader,
                   valid_loader, optimizer,
                   device, logging_interval=50,
                   verbose=1,
                   scheduler=None,
                   scheduler_on='valid_acc',
                   which_model='categorical',
                   with_embedding=False):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):


            if which_model != 'categorical':
                levels = levels_from_labelbatch(targets,
                                                num_classes=model.num_classes)
            
            if with_embedding:
                features, residues = features
                features = features.to(device)
                residues = residues.to(device)
                targets = targets.to(device)
                
                if which_model != 'categorical':
                    logits, probas = model(features, residues)
                else:
                    logits = model(features, residues)
                
            else:
                features = features.to(device)
                targets = targets.to(device)   
                if which_model == 'categorical':
                    logits = model(features)
                else:
                    logits, probas = model(features, residues)
            
            if which_model == 'niu':
                loss = niu_loss(logits, levels)
                
            elif which_model == 'coral':
                loss = coral_loss(logits, levels)
                
            elif which_model == 'categorical':
                loss = torch.nn.functional.cross_entropy(logits, targets)
                
            else:
                raise ValueError('This which_model choice is not supported.')
                
                

            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            minibatch_loss_list.append(loss.item())
            
            if verbose:
                if not batch_idx % logging_interval:
                    print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                          f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                          f'| Loss: {loss:.4f}')
 
        model.eval()
        with torch.no_grad():  # save memory during inference
            train_acc = compute_accuracy(
                model, train_loader, device=device,
                with_embedding=with_embedding,
                which_model=which_model)
            valid_acc = compute_accuracy(
                model, valid_loader, device=device,
                with_embedding=with_embedding,
                which_model=which_model)
            
            if verbose:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Train: {train_acc :.2f}% '
                      f'| Validation: {valid_acc :.2f}%')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        elapsed = (time.time() - start_time)/60
        if verbose:
            print(f'Time elapsed: {elapsed:.2f} min')
        
        if scheduler is not None:

            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(minibatch_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')
        

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    return minibatch_loss_list, train_acc_list, valid_acc_list
import os
from helper_evaluate import compute_accuracy, compute_mae_and_mse
from helper_evaluate import compute_epoch_loss
from helper_losses import niu_loss, coral_loss, conditional_loss
from helper_data import levels_from_labelbatch

import time
import torch
import torch.nn.functional as F

from collections import OrderedDict
import json
import subprocess
import sys
import numpy as np

def iteration_logging(info_dict, batch_idx,
                      loss, train_dataset,
                      frequency, epoch):

    logfile = info_dict['settings']['training logfile']
    batch_size = info_dict['settings']['batch size']
    num_epochs = info_dict['settings']['num epochs']

    info_dict['training']['minibatch loss'].append(loss.item())
    if not batch_idx % frequency:
        s = (f'Epoch: {epoch:03d}/{num_epochs:03d} | '
             f'Batch {batch_idx:04d}/'
             f'{len(train_dataset)//batch_size:04d} | '
             f'Loss: {loss:.4f}')
        print(s)
        with open(logfile, 'a') as f:
            f.write(f'{s}\n')


def epoch_logging(info_dict, model, loss, epoch,
                  start_time, which_model,
                  train_loader, valid_loader,
                  skip_train_eval=False):
    device = torch.device('cpu')
    path = info_dict['settings']['output path']
    logfile = info_dict['settings']['training logfile']


    model.eval()
    with torch.no_grad():  # save memory during inference
        train_acc = compute_accuracy(
            model, train_loader, device=device,
            which_model=which_model)
        valid_acc = compute_accuracy(
            model, valid_loader, device=device,
            which_model=which_model)
        train_mae,train_mse = compute_mae_and_mse(model, train_loader, device=device, which_model=which_model, 
                    with_embedding=False)
        valid_mae,valid_mse = compute_mae_and_mse(model, valid_loader, device=device, which_model=which_model, 
                    with_embedding=False)
        valid_rmse = torch.sqrt(valid_mse)
        train_rmse = torch.sqrt(train_mse)
        info_dict['training']['epoch train mae'].append(train_mae.item())
        info_dict['training']['epoch train acc'].append(train_acc.item())
        info_dict['training']['epoch train rmse'].append(train_rmse.item())
        info_dict['training']['epoch valid mae'].append(valid_mae.item())
        info_dict['training']['epoch valid acc'].append(valid_acc.item())
        info_dict['training']['epoch valid rmse'].append(valid_rmse.item())

        # if valid_mae < best_mae:
        #     best_mae, best_mse, best_epoch = valid_mae, valid_mse, epoch
        #     torch.save(model.state_dict(), os.path.join(PATH,'best_model.pt'))

        if valid_rmse < info_dict['training']['best running rmse']:
            info_dict['training']['best running mae'] = valid_mae.item()
            info_dict['training']['best running rmse'] = valid_rmse.item()
            info_dict['training']['best running acc'] = valid_acc.item()
            info_dict['training']['best running epoch'] = epoch
            # ######### SAVE MODEL #############
            torch.save(model.state_dict(), os.path.join(path, 'best_model.pt'))

        s = (f'MAE/RMSE/ACC: Current Valid: {valid_mae:.2f}/'
             f'{valid_rmse:.2f}/'
             f'{info_dict["training"]["best running acc"]:.2f} Ep. {epoch} |'
             f' Best Valid: {info_dict["training"]["best running mae"]:.2f}'
             f'/{info_dict["training"]["best running rmse"]:.2f}'
             f'/{info_dict["training"]["best running acc"]:.2f}'
             f' | Ep. {info_dict["training"]["best running epoch"]}')
        print(s)
        with open(logfile, 'a') as f:
            f.write('%s\n' % s)

        s = f'Time elapsed: {(time.time() - start_time)/60:.2f} min'
        print(s)
        with open(logfile, 'a') as f:
            f.write(f'{s}\n')
        # if verbose:
        #     print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
        #           f'| Train: {train_acc :.2f}% '
        #           f'| Validation: {valid_acc :.2f}%')
        #     print('MAE/RMSE: | Current Valid: %.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f Ep. %d' % (
        #         valid_mae, valid_mse, epoch, best_mae, best_mse, best_epoch))
        # train_acc_list.append(train_acc.item())
        # valid_acc_list.append(valid_acc.item())

    # elapsed = (time.time() - start_time)/60
    # if verbose:
    #     print(f'Time elapsed: {elapsed:.2f} min')
        

    

def aftertraining_logging(model, which, info_dict, train_loader,
                          valid_loader, test_loader, which_model,
                          start_time=None):

    device = torch.device('cpu')
    path = info_dict['settings']['output path']
    logfile = info_dict['settings']['training logfile']

    # if which == 'last':
    #     torch.save(model.state_dict(), os.path.join(path, 'last_model.pt'))
    #     info_dict_key = 'last'
    #     log_key = ''

    # elif which == 'best':
    #     model.load_state_dict(torch.load(os.path.join(path, 'best_model.pt')))
    #     info_dict_key = 'best'
    #     log_key = 'Best '

    # else:
    #     raise ValueError('`which` must be "last" or "best"')

    # elapsed = (time.time() - start_time)/60
    # print(f'Total Training Time: {elapsed:.2f} min')

    model.load_state_dict(torch.load(os.path.join(path, 'best_model.pt')))
    info_dict_key = 'best'
    log_key = 'Best '
    model.eval()

    with torch.set_grad_enabled(False):
        train_mae, train_mse = compute_mae_and_mse(model, train_loader, device=device, which_model=which_model, 
                        with_embedding=False)
        valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader, device=device, which_model=which_model, 
                        with_embedding=False)
        test_mae, test_mse = compute_mae_and_mse(model, test_loader, device=device, which_model=which_model, 
                        with_embedding=False)
        train_rmse, valid_rmse = torch.sqrt(train_mse), torch.sqrt(valid_mse)
        test_rmse = torch.sqrt(test_mse)

        # s = 'MAE/RMSE: | Best Train: %.2f/%.2f | Best Valid: %.2f/%.2f | Best Test: %.2f/%.2f' % (
        #     train_mae, test_mse,
        #     valid_mae, valid_mse,
        #     test_mae, test_mse)

        s = (f'MAE/RMSE: | {log_key}Train: {train_mae:.2f}/{train_rmse:.2f} '
             f'| {log_key}Valid: {valid_mae:.2f}/{valid_rmse:.2f} '
             f'| {log_key}Test: {test_mae:.2f}/{test_rmse:.2f}')
        print(s)
        with open(logfile, 'a') as f:
            f.write(f'{s}\n')

        train_acc = compute_accuracy(model, train_loader,
                                     device=device,
                                     which_model=which_model)
        valid_acc = compute_accuracy(model, valid_loader,
                                     device=device,
                                     which_model=which_model)
        test_acc = compute_accuracy(model, test_loader,
                                    device=device,
                                    which_model=which_model)

        s = (f'ACC: | {log_key}Train: {train_acc:.2f} '
             f'| {log_key}Valid: {valid_acc:.2f} '
             f'| {log_key}Test: {test_acc:.2f}')
        print(s)
        with open(logfile, 'a') as f:
            f.write(f'{s}\n')
        if start_time is not None:
            s = f'Total Running Time: {(time.time() - start_time)/60:.2f} min'
            print(s)
            with open(logfile, 'a') as f:
                f.write(f'{s}\n')
        info_dict[info_dict_key]['train mae'] = train_mae.item()
        info_dict[info_dict_key]['train rmse'] = train_rmse.item()
        info_dict[info_dict_key]['train acc'] = train_acc.item()
        info_dict[info_dict_key]['valid mae'] = valid_mae.item()
        info_dict[info_dict_key]['valid rmse'] = valid_rmse.item()
        info_dict[info_dict_key]['valid acc'] = train_acc.item()
        info_dict[info_dict_key]['test mae'] = test_mae.item()
        info_dict[info_dict_key]['test rmse'] = test_rmse.item()
        info_dict[info_dict_key]['test acc'] = test_acc.item()
        # print(s)

    # model.eval()
    # with torch.no_grad():
    #     train_mae, train_mse = compute_mae_and_mse(model, train_loader,
    #                                                device=device,
    #                                                which_model=which_model)
    #     valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
    #                                                device=device,
    #                                                which_model=which_model)
    #     test_mae, test_mse = compute_mae_and_mse(model, test_loader,
    #                                              device=device,
    #                                              which_model=which_model)
    #     train_rmse, valid_rmse = torch.sqrt(train_mse), torch.sqrt(valid_mse)
    #     test_rmse = torch.sqrt(test_mse)

    #     train_acc = compute_accuracy(model, train_loader,
    #                                  device=device,
    #                                  which_model=which_model)
    #     valid_acc = compute_accuracy(model, valid_loader,
    #                                  device=device,
    #                                  which_model=which_model)
    #     test_acc = compute_accuracy(model, test_loader,
    #                                 device=device,
    #                                 which_model=which_model)

        # s = (f'MAE/RMSE: | {log_key}Train: {train_mae:.2f}/{train_rmse:.2f} '
        #      f'| {log_key}Valid: {valid_mae:.2f}/{valid_rmse:.2f} '
        #      f'| {log_key}Test: {test_mae:.2f}/{test_rmse:.2f}')
        # print(s)
        # with open(logfile, 'a') as f:
        #     f.write(f'{s}\n')

        # s = (f'ACC: | {log_key}Train: {train_acc:.2f} '
        #      f'| {log_key}Valid: {valid_acc:.2f} '
        #      f'| {log_key}Test: {test_acc:.2f}')
        # print(s)
        # with open(logfile, 'a') as f:
        #     f.write(f'{s}\n')

        # if start_time is not None:
        #     s = f'Total Running Time: {(time.time() - start_time)/60:.2f} min'
        #     print(s)
        #     with open(logfile, 'a') as f:
        #         f.write(f'{s}\n')

        # info_dict[info_dict_key]['train mae'] = train_mae.item()
        # info_dict[info_dict_key]['train rmse'] = train_rmse.item()
        # info_dict[info_dict_key]['train acc'] = train_acc.item()
        # info_dict[info_dict_key]['valid mae'] = valid_mae.item()
        # info_dict[info_dict_key]['valid rmse'] = valid_rmse.item()
        # info_dict[info_dict_key]['valid acc'] = train_acc.item()
        # info_dict[info_dict_key]['test mae'] = test_mae.item()
        # info_dict[info_dict_key]['test rmse'] = test_rmse.item()
        # info_dict[info_dict_key]['test acc'] = test_acc.item()


def create_logfile(info_dict):
    header = []
    header.append(f'This script: {info_dict["settings"]["script"]}')
    header.append(f'PyTorch Version: {info_dict["settings"]["pytorch version"]}')
    # header.append(f'CUDA device: {info_dict["settings"]["cuda device"]}')
    # header.append(f'CUDA version: {info_dict["settings"]["cuda version"]}')
    header.append(f'Random seed: {info_dict["settings"]["random seed"]}')
    header.append(f'Learning rate: {info_dict["settings"]["learning rate"]}')
    header.append(f'Epochs: {info_dict["settings"]["num epochs"]}')
    header.append(f'Batch size: {info_dict["settings"]["batch size"]}')
    #header.append(f'Number of classes: {info_dict["settings"]["num classes"]}')
    header.append(f'Output path: {info_dict["settings"]["output path"]}')

    with open(info_dict["settings"]["training logfile"], 'w') as f:
        for entry in header:
            print(entry)
            f.write(f'{entry}\n')
            f.flush()




def train_model_v2(model, num_epochs, train_loader,
                   valid_loader, test_loader, optimizer, info_dict,
                   device, logging_interval=50,
                   verbose=1,
                   scheduler=None,
                   scheduler_on='valid_acc',
                   which_model='categorical',
                   with_embedding=False):

    start_time = time.time()
    
    if which_model == 'coral':
        PATH = '/Users/xintongshi/Desktop/GitHub/ordinal-conditional/src/models/mlp-for-tabular/runs/coral'
    elif which_model == 'niu':
        PATH = '/Users/xintongshi/Desktop/GitHub/ordinal-conditional/src/models/mlp-for-tabular/runs/niu'
    elif which_model == 'conditional':
        PATH = '/Users/xintongshi/Desktop/GitHub/ordinal-conditional/src/models/mlp-for-tabular/runs/conditional'
    else:
        PATH = '/Users/xintongshi/Desktop/GitHub/ordinal-conditional/src/models/mlp-for-tabular/runs/xentr'

    best_mae, best_mse, best_epoch = 999, 999, -1
    info_dict['training'] = {
           'num epochs': num_epochs,
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

    for epoch in range(1,num_epochs+1):

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
                    logits, probas = model(features)
            
            if which_model == 'niu':
                loss = niu_loss(logits, levels)
                
            elif which_model == 'coral':
                loss = coral_loss(logits, levels)
                
            elif which_model == 'categorical':
                loss = torch.nn.functional.cross_entropy(logits, targets)
            elif which_model == 'conditional':
                loss = conditional_loss(logits, targets, num_classes=model.num_classes)
                
            else:
                raise ValueError('This which_model choice is not supported.')
                
                

            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            iteration_logging(info_dict=info_dict, batch_idx=batch_idx,
                            loss=loss, train_dataset=train_loader.dataset,
                            frequency=50, epoch=epoch)
            
            # if verbose:
            #     if not batch_idx % logging_interval:
            #         print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
            #               f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
            #               f'| Loss: {loss:.4f}')
        #epoch logging
        epoch_logging(info_dict=info_dict,
                               model=model, train_loader=train_loader,
                               valid_loader=valid_loader,
                               which_model=which_model,
                               loss=loss, epoch=epoch, start_time=start_time)

        # model.eval()
        # with torch.no_grad():  # save memory during inference
        #     train_acc = compute_accuracy(
        #         model, train_loader, device=device,
        #         with_embedding=with_embedding,
        #         which_model=which_model)
        #     valid_acc = compute_accuracy(
        #         model, valid_loader, device=device,
        #         with_embedding=with_embedding,
        #         which_model=which_model)
        #     train_mae,train_mse = compute_mae_and_mse(model, train_loader, device=device, which_model=which_model, 
        #                 with_embedding=with_embedding)
        #     valid_mae,valid_mse = compute_mae_and_mse(model, valid_loader, device=device, which_model=which_model, 
        #                 with_embedding=with_embedding)

        #     if valid_mae < best_mae:
        #         best_mae, best_mse, best_epoch = valid_mae, valid_mse, epoch
        #         torch.save(model.state_dict(), os.path.join(PATH,'best_model.pt'))
        #     if verbose:
        #         print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
        #               f'| Train: {train_acc :.2f}% '
        #               f'| Validation: {valid_acc :.2f}%')
        #         print('MAE/RMSE: | Current Valid: %.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f Ep. %d' % (
        #             valid_mae, valid_mse, epoch, best_mae, best_mse, best_epoch))
        #     train_acc_list.append(train_acc.item())
        #     valid_acc_list.append(valid_acc.item())

        # elapsed = (time.time() - start_time)/60
        # if verbose:
        #     print(f'Time elapsed: {elapsed:.2f} min')
        
        if scheduler is not None:

            # if scheduler_on == 'valid_acc':
            #     scheduler.step(valid_acc_list[-1])
            # elif scheduler_on == 'minibatch_loss':
            #     scheduler.step(minibatch_loss_list[-1])
            # else:
            #     raise ValueError(f'Invalid `scheduler_on` choice.')
            scheduler.step(info_dict['training']['epoch valid rmse'][-1])

    #after training logging
    # elapsed = (time.time() - start_time)/60
    # print(f'Total Training Time: {elapsed:.2f} min')

    # model.load_state_dict(torch.load(os.path.join(PATH, 'best_model.pt')))
    # model.eval()

    # with torch.set_grad_enabled(False):
    #     train_mae, train_mse = compute_mae_and_mse(model, train_loader, device=device, which_model=which_model, 
    #                     with_embedding=with_embedding)
    #     valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader, device=device, which_model=which_model, 
    #                     with_embedding=with_embedding)
    #     test_mae, test_mse = compute_mae_and_mse(model, test_loader, device=device, which_model=which_model, 
    #                     with_embedding=with_embedding)

    #     s = 'MAE/RMSE: | Best Train: %.2f/%.2f | Best Valid: %.2f/%.2f | Best Test: %.2f/%.2f' % (
    #         train_mae, test_mse,
    #         valid_mae, valid_mse,
    #         test_mae, test_mse)
    #     print(s)
    info_dict['best'] = {}
    aftertraining_logging(model=model, which='best', info_dict=info_dict,
                        train_loader=train_loader,
                        valid_loader=valid_loader, test_loader=test_loader,
                        which_model=which_model,
                        start_time=start_time)
    return 
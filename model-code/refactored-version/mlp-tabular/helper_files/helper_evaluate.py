import torch
import torch.nn.functional as F
import numpy as np
from itertools import product

# from local helper files
from helper_data import label_to_levels, levels_from_labelbatch, proba_to_label
from helper_losses import niu_loss
from helper_losses import coral_loss


def get_labels_and_predictions(model, data_loader, device, which_model='categorical', with_embedding=False):
    model.eval()
    labels = []
    predictions = []
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

            if which_model == 'ordinal':
                levels = levels_from_labelbatch(targets,
                                                num_classes=model.num_classes)

            if with_embedding:
                features, residues = features
                features = features.to(device)
                residues = residues.to(device)
                targets = targets.to(device)
                
                if which_model != 'categorical':
                    logits, probas = model(features, residues)
                    predicted_labels = proba_to_label(probas).float()
                else:
                    logits = model(features, residues)
                    _, predicted_labels = torch.max(logits, 1)
            else:
                features = features.to(device)
                targets = targets.to(device)   
                if which_model != 'categorical':
                    logits, probas = model(features)
                    predicted_labels = proba_to_label(probas).float()
                else:
                    logits = model(features)
                    _, predicted_labels = torch.max(logits, 1)

            labels.append(targets)
            predictions.append(predicted_labels)

    return torch.cat(labels), torch.cat(predictions)


def compute_accuracy(model, data_loader, device, which_model='categorical', with_embedding=False):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):
            
            if which_model == 'ordinal':
                levels = levels_from_labelbatch(targets,
                                                num_classes=model.num_classes)
            
            if with_embedding:
                features, residues = features
                features = features.to(device)
                residues = residues.to(device)
                targets = targets.to(device)
                
                if which_model != 'categorical':
                    logits, probas = model(features, residues)
                    predicted_labels = proba_to_label(probas).float()
                else:
                    logits = model(features, residues)
                    _, predicted_labels = torch.max(logits, 1)
            else:
                features = features.to(device)
                targets = targets.to(device)   
                if which_model != 'categorical':
                    logits, probas = model(features)
                    predicted_labels = proba_to_label(probas).float()
                else:
                    logits = model(features)
                    _, predicted_labels = torch.max(logits, 1)
            
            #if isinstance(logits, torch.distributed.rpc.api.RRef):
            #    logits = logits.local_value()

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def compute_mae_and_mse(model, data_loader, device, which_model, 
                        with_embedding=False):

    with torch.no_grad():

        mae, mse, num_examples = 0., 0., 0

        for i, (features, targets) in enumerate(data_loader):
            
            if which_model == 'ordinal':
                levels = levels_from_labelbatch(targets,
                                                num_classes=model.num_classes)

            if with_embedding:
                features, residues = features
                features = features.to(device)
                residues = residues.to(device)
                targets = targets.to(device)
                
                if which_model != 'categorical':
                    logits, probas = model(features, residues)
                    predicted_labels = proba_to_label(probas).float()
                else:
                    logits = model(features, residues)
                    _, predicted_labels = torch.max(logits, 1)
            else:
                features = features.to(device)
                targets = targets.to(device)   
                if which_model != 'categorical':
                    logits, probas = model(features)
                    # _, predicted_labels = torch.max(probas, 1)
                    predicted_labels = proba_to_label(probas).float()
                else:
                    logits = model(features)
                    _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(predicted_labels - targets))
            mse += torch.sum((predicted_labels - targets)**2)

        mae = mae / num_examples
        mse = mse / num_examples
        return mae, mse



def compute_epoch_loss(model, data_loader, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def compute_confusion_matrix(model, data_loader, device):

    all_targets, all_predictions = [], []
    with torch.no_grad():

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            all_targets.extend(targets.to('cpu'))
            all_predictions.extend(predicted_labels.to('cpu'))

    all_predictions = all_predictions
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    class_labels = np.unique(np.concatenate((all_targets, all_predictions)))
    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])
    n_labels = class_labels.shape[0]
    lst = []
    z = list(zip(all_targets, all_predictions))
    for combi in product(class_labels, repeat=2):
        lst.append(z.count(combi))
    mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)
    return mat
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_training_loss(info_dict, averaging_iterations=100):

    path = info_dict['settings']['output path']

    num_epochs = info_dict['training']['num epochs']
    #  epoch_train_mae = info_dict['training']['epoch train mae']
    #  epoch_valid_mae = info_dict['training']['epoch valid mae']
    minibatch_loss = info_dict['training']['minibatch loss']

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_loss)),
             (minibatch_loss), label='Minibatch Loss')

    if len(minibatch_loss) < 1000:
        num_losses = len(minibatch_loss) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([
        0, np.max(minibatch_loss[num_losses:])*1.5
        ])
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    ax1.plot(np.convolve(minibatch_loss,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label='Running Average')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*info_dict['training']['iter per epoch'] for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()

    plt.savefig(os.path.join(path, 'plot_training_loss.pdf'))
    plt.clf()


def plot_mae(info_dict):

    path = info_dict['settings']['output path']
    num_epochs = info_dict['training']['num epochs']

    if len(info_dict['training']['epoch train mae']):
        plt.plot(np.arange(1, num_epochs+1),
                 info_dict['training']['epoch train mae'], label='Training')
    if len(info_dict['training']['epoch valid mae']):
        plt.plot(np.arange(1, num_epochs+1),
                 info_dict['training']['epoch valid mae'], label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'plot_mae_training_validation.pdf'))
    plt.clf()


def plot_accuracy(info_dict):

    path = info_dict['settings']['output path']
    num_epochs = info_dict['training']['num epochs']

    if len(info_dict['training']['epoch train acc']):
        plt.plot(np.arange(1, num_epochs+1),
                 info_dict['training']['epoch train acc'], label='Training')
    if len(info_dict['training']['epoch valid acc']):    
        plt.plot(np.arange(1, num_epochs+1),
                 info_dict['training']['epoch valid acc'], label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'plot_acc_training_validation.pdf'))
    plt.clf()


def plot_per_class_mae(info_dict):

    path = info_dict['settings']['output path']

    for best_or_last in ('best', 'last'):
        for train_or_test in ('train', 'test'):        

            errors = info_dict[f'per-class mae {train_or_test} ({best_or_last} model)']
            #actual_selfentropy = info_dict['test set mae self-entropy']
            #ideal_selfentropy = info_dict['ideal test set mae self-entropy']

            plt.bar(range(len(errors)), errors)
            #plt.title(f'\nSelf Entropy: {actual_selfentropy:.2f}'
            #          f'\nIdeal Self Entropy: {ideal_selfentropy:.2f}')
            plt.xlabel('Label')
            plt.ylabel('MAE')
            plt.savefig(os.path.join(path,
                        f'per-class-mae_{train_or_test}-set_{best_or_last}-model.pdf'))
            plt.clf()

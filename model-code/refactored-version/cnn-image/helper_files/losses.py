import torch.nn.functional as F
import torch


def coral_loss(logits, levels, importance_weights=None, reduction='mean'):
    """Computes the CORAL loss described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        Outputs of the CORAL layer.

    levels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors
        (via `coral_pytorch.dataset.levels_from_labelbatch`).

    importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
        Optional weights for the different labels in levels.
        A tensor of ones, i.e.,
        `torch.ones(num_classes-1, dtype=torch.float32)`
        will result in uniform weights that have the same effect as None.

    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value
        (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).

    Examples
    ----------
    >>> import torch
    >>> levels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> coral_loss(logits, levels)
    tensor(0.6920)
    """

    if not logits.shape == levels.shape:
        raise ValueError(f"Please ensure that logits ({logits.shape})"
                         f" has the same shape as levels ({levels.shape}). ")

    term1 = (F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))

    if importance_weights is not None:
        term1 *= importance_weights

    val = (-torch.sum(term1, dim=1))

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss


def niu_loss(logits, levels):
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1]*levels
                      + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels)), dim=1))
    return torch.mean(val)


def loss_conditional(logits, y_train, NUM_CLASSES):
    sets = []
    for i in range(NUM_CLASSES-1):
        label_mask = y_train > i-1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    losses = 0
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]
        pred = logits[train_examples, task_index]
        if len(s[1]) < 1:
            continue
        loss = -torch.mean( F.logsigmoid(pred)*train_labels
                            + (F.logsigmoid(pred) - pred)*(1-train_labels) )

        losses += loss

    return losses/len(sets)


def loss_conditional_v2(logits, y_train, NUM_CLASSES):
    """Compared to the previous conditional loss, here, the loss is computed 
       as the average loss of the total samples, instead of firstly averaging 
       the cross entropy inside each task and then averaging over tasks equally. 
    """
    sets = []
    for i in range(NUM_CLASSES-1):
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
        
        loss = -torch.sum( F.logsigmoid(pred)*train_labels
                                + (F.logsigmoid(pred) - pred)*(1-train_labels) )
        losses += loss
    return losses/num_examples


def loss_conditional_v2_ablation(logits, y_train, NUM_CLASSES):
    """Same as loss_conditional_v2 but without training subsets
    to faciliate an ablation study.
    """
    sets = []
    for i in range(NUM_CLASSES-1):
        i = 0  # this basically eliminates subsets. 
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
        
        loss = -torch.sum( F.logsigmoid(pred)*train_labels
                                + (F.logsigmoid(pred) - pred)*(1-train_labels) )
        losses += loss
    return losses/num_examples
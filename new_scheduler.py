from torch.optim.lr_scheduler import LRScheduler
from data_loader import get_loader4scheduler
import torch
from torch import nn
import numpy as np
import random
import copy
import warnings


def weighted_geometric_mean(values, weights):
    normalized_weights = weights / torch.sum(weights)
    log_values = torch.log(values)
    weighted_log_sum = torch.sum(normalized_weights * log_values)
    geometric_mean = torch.exp(weighted_log_sum)
    
    return geometric_mean



class loss_informed_scheduler(LRScheduler):
    def __init__(self, optimizer, model, factors=[2, 1, 0.5],total_iters=5, last_epoch=-1,
                 verbose="deprecated", batch_size=128, initial_lr = 0.05):

        self.total_iters = total_iters
        self.factors = factors
        self.model = model
        self.loader = get_loader4scheduler(batch_size=batch_size)
        random.seed(0)
        self._last_lr = torch.tensor([initial_lr])
        self.initialize = True
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.initialize: 
            self.initialize = False
            return [self._last_lr[0] for _ in self.optimizer.param_groups]
        models = [copy.deepcopy(self.model) for _ in range(len(self.factors))]
        models = list(zip(models, self.factors))
        losses = []
        prior_lr = self.get_last_lr()[0]
        print("!!!! loss informed scheduler")
        criterion = nn.CrossEntropyLoss()
        for temp_model, factor in models:
            temp_optimizer = torch.optim.SGD(temp_model.parameters(), lr=prior_lr * factor)
            train_loss = 0
            random_index = random.randint(0, len(self.loader) - 1)
            images = None
            labels = None
            for i, batch in enumerate(self.loader):
                if i == random_index:
                    images, labels = batch 
            if(torch.cuda.is_available()):
                images, labels = images.cuda(), labels.cuda()
            temp_optimizer.zero_grad()
            output = temp_model(images)
            loss = criterion(output, labels)
            loss.backward()
            temp_optimizer.step()
            train_loss += loss.item()
            losses.append(train_loss)
        losses = torch.tensor(losses, dtype=torch.float32)
        normalized_losses = ((losses - torch.mean(losses)) / torch.std(losses))
        scores = torch.exp(-normalized_losses)
        factors = torch.tensor(self.factors, dtype=torch.float32)
        result_factor = weighted_geometric_mean(factors, scores)
        print(f"{factors}, {losses}, {scores}, {result_factor}")
        self._last_lr = [group['lr'] * result_factor for group in self.optimizer.param_groups]
        return self._last_lr
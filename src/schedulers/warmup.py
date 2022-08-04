from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, verbose=False, max_lr=3.5e-4, num_epochs=70):
        self.max_lr = max_lr
        self.num_epochs = num_epochs
        super(WarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):

        if self.last_epoch <= 10:
            return [self.max_lr * (self.last_epoch / 10)] * len(self.optimizer.param_groups)

        elif self.num_epochs / 7 < self.last_epoch <= self.num_epochs * 4 / 7:
            return [self.max_lr] * len(self.optimizer.param_groups)

        elif self.num_epochs * 4 / 7 < self.last_epoch <= self.num_epochs:
            return [self.max_lr / 10] * len(self.optimizer.param_groups)

        elif self.num_epochs < self.last_epoch:
            return [self.max_lr / 100] * len(self.optimizer.param_groups)

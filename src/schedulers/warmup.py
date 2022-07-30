from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super(WarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):

        if self.last_epoch <= 10:
            return [3.5e-4 * (self.last_epoch / 10) for group in self.optimizer.param_groups]

        elif 10 < self.last_epoch <= 40:
            return [3.5e-4 for group in self.optimizer.param_groups]

        elif 40 < self.last_epoch <= 70:
            return [3.5e-5 for group in self.optimizer.param_groups]

        elif 70 < self.last_epoch:
            return [3.5e-6 for group in self.optimizer.param_groups]

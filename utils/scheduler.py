# Editor       : pycharm
# File name    : utils/scheduler.py
# Author       : huangxinyu
# Created date : 2020-11-15
# Description  : 动态调整学习率

import numpy as np
from config.yolov3 import cfg

class adjust_lr_by_wave(object):
    def __init__(self,optimizer,iter_max,lr_start,lr_end=0.,warmup=0):
        self.optimizer = optimizer
        self.iter_max = iter_max
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.warmup = warmup

    def step(self,epoch,iter):
        if (epoch == 0) & (iter <= cfg.warmup):
            lr = 5e-4 * (iter / cfg.warmup) ** 2
        else:
            if epoch < 5:
                lr = 5e-4
            elif epoch < 10:
                lr = 1e-4
            elif epoch < 15:
                lr = 5e-5
            else:
                lr = 1e-5
        for g in self.optimizer.param_groups:
            g['lr'] = lr

class adjust_lr_by_loss(object):
    def __init__(self,optimizer,lr_start,warmup,iters_every_epoch):
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.warmup = warmup
        self.iters_every_epoch = iters_every_epoch
        self.original_loss = []
        self.lr_divide = lr_start

    def step(self,iter,loss):

        if (iter % self.iters_every_epoch)==0:
            if len(self.original_loss)<10:
                self.original_loss.append(loss)
            else:
                self.original_loss.pop(0)
                self.original_loss.append(loss)
                print(loss)
                print(sum(self.original_loss) / 10)
                if loss >= sum(self.original_loss) / 10:
                    self.lr_divide /= np.e
        lr = 0
        if iter < self.warmup and self.warmup:
            lr = self.lr_start / self.warmup * iter
        elif len(self.original_loss)==10:
                lr = self.lr_divide
        else:
            lr = self.lr_start
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
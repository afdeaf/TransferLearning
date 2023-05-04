# Copyright (c) 2022 Raven Stock. email:cquptriven@qq.com

def inv_lr_scheduler(optimizer, ite_rate, gamma=10.0, power=0.75, lr=0.0001):
    lr = lr * (1 + gamma * ite_rate) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']

    return lr


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']
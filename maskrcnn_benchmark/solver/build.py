# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import apex

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    syncbn_layer_names = set()
    for key, value in model.named_modules():
        if isinstance(value, apex.parallel.SyncBatchNorm):
            syncbn_layer_names.add(key)

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        bias_in_syncbn = False

        if "bias" in key:
            for syncbn_layer_name in syncbn_layer_names:
                if syncbn_layer_name in key:
                    bias_in_syncbn = True
                    break
            if bias_in_syncbn:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.SYNCBN_BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.SYNCBN_WEIGHT_DECAY_BIAS
            else:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )

import logging as log
from abc import ABC, abstractmethod

import torch
import math

from pytorch_lightning.callbacks import GradientAccumulationScheduler


class TrainerTrainingTricksMixin(ABC):

    def __init__(self):
        # this is just a summary on variables used in this abstract class,
        #  the proper values/initialisation should be done in child class
        self.gradient_clip_val = None

    @abstractmethod
    def get_model(self):
        # this is just empty shell for code from other class
        pass

    def clip_gradients(self):
        # this code is a modification of torch.nn.utils.clip_grad_norm_
        # with TPU support based on https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md
        if self.gradient_clip_val > 0:
            model = self.get_model()
            parameters = model.parameters()
            max_norm = self.gradient_clip_val
            norm_type = 2
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            parameters = list(filter(lambda p: p.grad is not None, parameters))
            max_norm = float(max_norm)
            norm_type = float(norm_type)
            if norm_type == math.inf:
                total_norm = max(p.grad.data.abs().max() for p in parameters)
            else:
                device = parameters[0].device
                total_norm = torch.zeros([], device=device if parameters else None)
                for p in parameters:
                    param_norm = p.grad.data.norm(norm_type) ** norm_type
                total_norm.add_(param_norm)
                total_norm = (total_norm ** (1. / norm_type))
            clip_coef = torch.tensor(max_norm, device=device) / (total_norm + 1e-6)
            for p in parameters:
                p.grad.data.mul_(torch.where(clip_coef < 1, clip_coef, torch.tensor(1., device=device)))

    def print_nan_gradients(self):
        model = self.get_model()
        for param in model.parameters():
            if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                log.info(param, param.grad)

    def configure_accumulated_gradients(self, accumulate_grad_batches):
        self.accumulate_grad_batches = None

        if isinstance(accumulate_grad_batches, dict):
            self.accumulation_scheduler = GradientAccumulationScheduler(accumulate_grad_batches)
        elif isinstance(accumulate_grad_batches, int):
            schedule = {1: accumulate_grad_batches}
            self.accumulation_scheduler = GradientAccumulationScheduler(schedule)
        else:
            raise TypeError("Gradient accumulation supports only int and dict types")

        self.accumulation_scheduler.set_trainer(self)

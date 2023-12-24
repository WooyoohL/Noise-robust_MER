import numpy as np
import torch
import math
from diffusers import DDPMScheduler
from torch import Tensor, Generator, strided, memory_format, contiguous_format, strided, inf
from typing import List, Tuple, Optional, Union, Any, ContextManager, Callable, overload, Iterator, NamedTuple, \
    Sequence, Literal, TypeVar

from torch.types import _int, _float, _bool, Number, _dtype, _device, _qscheme, _size, _layout, SymInt, Device
import torch

import builtins
import torch.nn as nn
from diffusers import SchedulerMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config


def add_missing(input, times):
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "scaled_linear"  # linear, scaled_linear, squaredcos_cap_v2
    num_train_time_steps: int = 10

    noise_scheduler = DDPMScheduler(
        beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule,
        num_train_timesteps=num_train_time_steps)
    # noise = torch.randn_like(input)

    bsz_input = 1
    time_steps = torch.randint(0, num_train_time_steps, (bsz_input,),
                               device=input.device)
    time_steps = time_steps.long()

    noisy_latents = input
    for i in range(times):  # times: 1-10
        # times = 10  ->  MSE_loss = 4     acc = 0.25
        # times = 5   ->  MSE_loss = 2.5   acc = 0.35
        # times = 4   ->  MSE_loss = 2     acc = 0.40
        # times = 3   ->  MSE_loss = 1.5
        # times = 2   ->  MSE_loss = 1.0
        # times = 1   ->  MSE_loss = 0.5
        noise = torch.randn_like(input)
        noisy_latents = noise_scheduler.add_noise(noisy_latents, noise, time_steps)

    return noisy_latents


def betas_for_alpha_bar(num_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class NoiseScheduler:
    def __init__(self,
                 beta_start: float = 0.01,  # 0.0001
                 beta_end: float = 0.5,  # 0.9 约100步
                 beta_schedule: str = "scaled_linear",  # linear, scaled_linear, squaredcos_cap_v2
                 num_time_steps: int = 100,
                 noise_type: str = 'Gaussian',  # default: 'Gaussian',  Optional: 'Impulse'
                 impu_times: int = 10,
                 occur_freq: float = 0.7  # the occurrence frequency of impulse noise, from 0.1 to 1.0
                 ):
        # scheduler config
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.num_time_steps = num_time_steps

        # styles and scales of noise
        self.noise_type = noise_type

        self.impu_times = impu_times
        # from 0 to 1, when use Impulse_Noise, the threshold can decide the scale of noise
        self.occur_freq = occur_freq



        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_time_steps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                    torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_time_steps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(self.num_time_steps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, self.num_time_steps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

    def add_noise(self, input, miss_index):
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = alphas_cumprod.to(device=input.device, dtype=input.dtype)
        time_steps = torch.randint(0, self.num_time_steps, (1,),  # 1 = batch_size
                                   device=input.device)
        time_steps = time_steps.long()
        sqrt_alpha_prod = alphas_cumprod[time_steps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(input.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[time_steps]) ** 0.5

        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(input.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        miss = [input[i] for i in range(len(input))]
        reverse = [input[i] for i in range(len(input))]

        for i in range(0, len(input)):
            latents = miss[i]
            noise = torch.randn_like(latents)
            if self.noise_type == 'Impulse':
                noise = torch.sign(noise)
                mask = torch.rand(latents.shape).cuda() > self.occur_freq
                mask = mask.to(input.device)
                noise = noise * mask.float()

            noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
            latents = latents.unsqueeze(0)
            if miss_index[i].item() == 0:
                miss[i] = noisy_latents
                reverse[i] = latents
            elif miss_index[i].item() == 1:
                miss[i] = latents
                reverse[i] = noisy_latents
            else:
                print(miss_index[i])
                print('error')

        miss = torch.cat(miss, dim=0)
        reverse = torch.cat(reverse, dim=0)
        return miss, reverse





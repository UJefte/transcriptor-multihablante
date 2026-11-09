#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class CLayerNorm(nn.LayerNorm):
    """Normalización de capa por canal."""

    def __init__(self, *args, **kwargs):
        super(CLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, sample):
        """Pasada hacia adelante."""
        if sample.dim() != 3:
            raise RuntimeError('{} only accept 3-D tensor as input'.format(
                self.__name__))
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        # normalización de capa
        sample = super().forward(sample)
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample

class ILayerNorm(nn.InstanceNorm1d):
    """Normalización de capa por canal."""

    def __init__(self, *args, **kwargs):
        super(ILayerNorm, self).__init__(*args, **kwargs)

    def forward(self, sample):
        """Pasada hacia adelante."""
        if sample.dim() != 3:
            raise RuntimeError('{} only accept 3-D tensor as input'.format(
                self.__name__))
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        # normalización de capa
        sample = super().forward(sample)
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample

class GLayerNorm(nn.Module):
    """Normalización de capa global para TasNet."""

    def __init__(self, channels, eps=1e-5):
        super(GLayerNorm, self).__init__()
        self.eps = eps
        self.norm_dim = channels
        self.gamma = nn.Parameter(torch.Tensor(channels))
        self.beta = nn.Parameter(torch.Tensor(channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, sample):
        """Pasada hacia adelante."""
        if sample.dim() != 3:
            raise RuntimeError('{} only accept 3-D tensor as input'.format(
                self.__name__))
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        # media y varianza [N, 1, 1]
        mean = torch.mean(sample, (1, 2), keepdim=True)
        var = torch.mean((sample - mean)**2, (1, 2), keepdim=True)
        sample = (sample - mean) / torch.sqrt(var + self.eps) * \
                self.gamma + self.beta
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample

class _LayerNorm(nn.Module):
    """Clase base de normalización de capa."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Espera una entrada de tamaño [batch, canal, *]."""
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLayerNorm(_LayerNorm):
    """Normalización de capa global (globLN)."""

    def forward(self, x):
        """Pasada hacia adelante; sirve para entradas de más de 2 dimensiones."""
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())

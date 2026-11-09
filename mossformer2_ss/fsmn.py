import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.nn.parameter import Parameter
import numpy as np
import os

class UniDeepFsmn(nn.Module):
    """Red FSMN de una dirección."""

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        
        # se crean las capas
        self.linear = nn.Linear(input_dim, hidden_size)  # capa lineal al tamaño oculto
        self.project = nn.Linear(hidden_size, output_dim, bias=False)  # del tamaño oculto al tamaño de salida
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder + lorder - 1, 1], [1, 1], groups=output_dim, bias=False)  # capa de convolución

    def forward(self, input):
        """Pasada por UniDeepFsmn."""
        f1 = F.relu(self.linear(input))  # capa lineal y después activación ReLU
        p1 = self.project(f1)  # se lleva al tamaño de salida
        x = th.unsqueeze(p1, 1)  # se agrega una dimensión porque Conv2d la necesita
        x_per = x.permute(0, 3, 2, 1)  # se reacomodan las dimensiones para la convolución
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])  # relleno para la convolución causal
        out = x_per + self.conv1(y)  # se suma la entrada original a la salida
        out1 = out.permute(0, 3, 2, 1)  # se regresan las dimensiones como estaban
        return input + out1.squeeze()  # regresa la entrada mejorada


class UniDeepFsmn_dual(nn.Module):
    """Red FSMN doble: agrega otra convolución para sacar más características."""

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn_dual, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        
        # se crean las capas
        self.linear = nn.Linear(input_dim, hidden_size)  # capa lineal al tamaño oculto
        self.project = nn.Linear(hidden_size, output_dim, bias=False)  # del tamaño oculto al tamaño de salida
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder + lorder - 1, 1], [1, 1], groups=output_dim, bias=False)  # primera convolución
        self.conv2 = nn.Conv2d(output_dim, output_dim, [lorder + lorder - 1, 1], [1, 1], groups=output_dim // 4, bias=False)  # segunda convolución

    def forward(self, input):
        """Pasada por UniDeepFsmn_dual."""
        f1 = F.relu(self.linear(input))  # capa lineal y después activación ReLU
        p1 = self.project(f1)  # se lleva al tamaño de salida
        x = th.unsqueeze(p1, 1)  # se agrega una dimensión porque Conv2d la necesita
        x_per = x.permute(0, 3, 2, 1)  # se reacomodan las dimensiones para la convolución
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])  # relleno para la convolución causal
        conv1_out = x_per + self.conv1(y)  # se suma la entrada original a la primera convolución
        z = F.pad(conv1_out, [0, 0, self.lorder - 1, self.lorder - 1])  # relleno para la segunda convolución
        out = conv1_out + self.conv2(z)  # se suma la salida de la segunda convolución
        out1 = out.permute(0, 3, 2, 1)  # se regresan las dimensiones como estaban
        return input + out1.squeeze()  # regresa la entrada mejorada


class DilatedDenseNet(nn.Module):
    """Red densa con convoluciones dilatadas: ve más contexto con pocos parámetros."""

    def __init__(self, depth=4, lorder=20, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)  # relleno de la entrada
        self.twidth = lorder * 2 - 1  # ancho del kernel
        self.kernel_size = (self.twidth, 1)  # tamaño del kernel de las convoluciones

        # se crean las capas según la profundidad
        for i in range(self.depth):
            dil = 2 ** i  # se calcula la dilatación
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1  # se calcula cuánto rellenar
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((0, 0, pad_length, pad_length), value=0.))  # relleno para la dilatación
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1), groups=self.in_channels, bias=False))  # convolución con dilatación
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))  # capa de normalización
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))  # capa de activación

    def forward(self, x):
        """Pasada por DilatedDenseNet."""
        skip = x  # empieza la conexión de salto
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)  # se aplica el relleno
            out = getattr(self, 'conv{}'.format(i + 1))(out)  # se aplica la convolución
            out = getattr(self, 'norm{}'.format(i + 1))(out)  # se aplica la normalización
            out = getattr(self, 'prelu{}'.format(i + 1))(out)  # se aplica la activación PReLU
            skip = th.cat([out, skip], dim=1)  # se junta la salida con la conexión de salto
        return out  # regresa la salida final

class UniDeepFsmn_dilated(nn.Module):
    """UniDeepFsmn combinada con una red densa dilatada."""

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None, depth=2):
        super(UniDeepFsmn_dilated, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        
        # se crean las capas
        self.linear = nn.Linear(input_dim, hidden_size)  # capa lineal al tamaño oculto
        self.project = nn.Linear(hidden_size, output_dim, bias=False)  # del tamaño oculto al tamaño de salida
        self.conv = DilatedDenseNet(depth=self.depth, lorder=lorder, in_channels=output_dim)  # red densa dilatada para sacar características

    def forward(self, input):
        """Pasada por UniDeepFsmn_dilated."""
        f1 = F.relu(self.linear(input))  # capa lineal y después activación ReLU
        p1 = self.project(f1)  # se lleva al tamaño de salida
        x = th.unsqueeze(p1, 1)  # se agrega una dimensión porque Conv2d la necesita
        x_per = x.permute(0, 3, 2, 1)  # se reacomodan las dimensiones para la convolución
        out = self.conv(x_per)  # pasa por la red densa dilatada
        out1 = out.permute(0, 3, 2, 1)  # se regresan las dimensiones como estaban

        return input + out1.squeeze()  # regresa la entrada mejorada

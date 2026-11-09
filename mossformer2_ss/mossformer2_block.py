"""Bloque MossFormer2.
Código reescrito por Shengkui Zhao a partir de https://github.com/lucidrains/FLASH-pytorch"""

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
# (se quitó "from torchinfo import summary": solo se usaba para imprimir el resumen del modelo)
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

from .conv_module import ConvModule, GLU, FFConvM_Dilated
from .fsmn import UniDeepFsmn, UniDeepFsmn_dilated
from .layer_norm import CLayerNorm, GLayerNorm, GlobLayerNorm, ILayerNorm

# funciones

def identity(t, *args, **kwargs):
    """Regresa la entrada sin cambios."""
    return t

def append_dims(x, num_dims):
    """Agrega dimensiones extra a x."""
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

def exists(val):
    """Dice si un valor existe (no es None)."""
    return val is not None

def default(val, d):
    """Regresa el valor si existe; si no, regresa d."""
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    """Relleno que falta para que n sea múltiplo de mult."""
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# capa ScaleNorm

class ScaleNorm(nn.Module):
    """ScaleNorm: variante de LayerNorm que escala la entrada."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5  # factor de escala
        self.eps = eps  # epsilon para evitar divisiones entre cero
        self.g = nn.Parameter(torch.ones(1))  # escala que se entrena

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale  # se calcula la norma
        return x / norm.clamp(min=self.eps) * self.g  # se escala la entrada con la norma

# codificación de posición absoluta

class ScaledSinuEmbedding(nn.Module):
    """Posiciones senoidales con una escala que se aprende."""
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))  # escala que se aprende
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # frecuencia inversa para los senos
        self.register_buffer('inv_freq', inv_freq)  # se guarda sin entrenarse

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device=device).type_as(self.inv_freq)  # índices de tiempo
        sinu = einsum('i, j -> i j', t, self.inv_freq)  # función senoidal
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)  # se juntan senos y cosenos
        return emb * self.scale  # se escalan las posiciones

# capa de desplazamiento y escala

class OffsetScale(nn.Module):
    """Aplica escala y desplazamiento a cada elemento."""
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))  # escala que se aprende
        self.beta = nn.Parameter(torch.zeros(heads, dim))  # desplazamiento que se aprende
        nn.init.normal_(self.gamma, std=0.02)  # gamma empieza con una desviación chica

    def forward(self, x):
        # se aplica escala y desplazamiento, y se parte por cabezas
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)  # se separa el tensor por cabezas

# módulo de convolución hacia adelante

class FFConvM(nn.Module):
    """Módulo con normalización, capa lineal y convolución."""
    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),  # se aplica la normalización
            nn.Linear(dim_in, dim_out),  # transformación lineal
            nn.SiLU(),  # activación SiLU
            ConvModule(dim_out),  # se aplica el módulo de convolución
            nn.Dropout(dropout)  # dropout para no sobreajustar
        )

    def forward(self, x):
        output = self.mdl(x)  # pasa por el módulo
        return output

class FFM(nn.Module):
    """Red básica con normalización, capa lineal, activación y dropout."""
    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        super().__init__()
        # red secuencial con normalización, capa lineal y activación
        self.mdl = nn.Sequential(
            norm_klass(dim_in),           # se normaliza para estabilizar el aprendizaje
            nn.Linear(dim_in, dim_out),   # capa lineal para llegar al tamaño de salida
            nn.SiLU(),                    # activación SiLU
            nn.Dropout(dropout)           # dropout para no sobreajustar
        )

    def forward(self, x):
        """Pasada por la red."""
        output = self.mdl(x)  # se aplica el módulo a la entrada
        return output

class FLASH_ShareA_FFConvM(nn.Module):
    """Bloque que combina FFConvM con atención por grupos."""
    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=1.,
        causal=False,
        dropout=0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)        
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # posiciones rotatorias
        self.rotary_pos_emb = rotary_pos_emb

        # capa de dropout
        self.dropout = nn.Dropout(dropout)

        # proyecciones de la entrada
        self.to_hidden = FFConvM(  # FFConvM para el valor y la compuerta
            dim_in=dim,
            dim_out=hidden_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.to_qk = FFConvM(  # FFConvM para consulta y llave
            dim_in=dim,
            dim_out=query_key_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )

        # escala y desplazamiento para la atención
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)

        # proyección de salida
        self.to_out = FFConvM(  # FFConvM para juntar todo y dar la salida final
            dim_in=dim*2,
            dim_out=dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        
        # compuerta sigmoide
        self.gateActivate = nn.Sigmoid()

    def forward(self, x, *, mask=None):
        """Pasada por el bloque."""

        # se guarda la entrada para la conexión de salto
        residual = x

        # desplazamiento de tokens opcional
        if self.shift_tokens:
            x_shift, x_pass = x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)  # se desplazan los tokens
            x = torch.cat((x_shift, x_pass), dim=-1)

        # proyecciones del valor y la compuerta
        v, u = self.to_hidden(x).chunk(2, dim=-1)  # se parte en dos ramas: v y u
        qk = self.to_qk(x)  # proyecciones de consulta y llave

        # escala y desplazamiento de consultas y llaves
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)

        # se calcula la atención
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, mask)

        # se combina la atención con la compuerta
        out = (att_u * v) * self.gateActivate(att_v * u)

        # conexión residual y proyección de salida
        x = residual + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        """Calcula la atención cuadrática y la lineal."""
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        # máscara para la atención lineal (si la hay)
        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # posiciones rotatorias de consultas y llaves (si las hay)
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # relleno para completar el grupo
        padding = padding_to_multiple_of(n, g)
        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: F.pad(t, (0, 0, 0, padding), value=0.), (quad_q, quad_k, lin_q, lin_k, v, u))

            mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
            mask = F.pad(mask, (0, padding), value=False)

        # se agrupa la entrada a lo largo de la secuencia
        quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n=self.group_size), (quad_q, quad_k, lin_q, lin_k, v, u))

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j=g)

        # atención cuadrática
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g
        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        # atención lineal (con suma acumulada en modo causal)
        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g
            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / g
            lin_ku = lin_ku.cumsum(dim=1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        # se regresan los grupos a la secuencia completa
        return map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out_v + lin_out_v, quad_out_u + lin_out_u))

class Gated_FSMN(nn.Module):
    """Bloque FSMN con compuerta: junta dos FFConvM."""
    def __init__(
        self,
        in_channels,
        out_channels,
        lorder,
        hidden_size
    ):
        super().__init__()

        # bloque FFConvM de la rama 'u'
        self.to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )

        # bloque FFConvM de la rama 'v'
        self.to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )

        # FSMN de una dirección (UniDeepFsmn) para la rama 'u'
        self.fsmn = UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)

    def forward(self, x):
        """Pasada por Gated_FSMN."""
        input = x  # se guarda la entrada para la conexión de salto

        # la entrada pasa por FFConvM en las ramas 'u' y 'v'
        x_u = self.to_u(x)
        x_v = self.to_v(x)

        # se aplica FSMN a la rama 'u'
        x_u = self.fsmn(x_u)

        # se combinan 'u' y 'v' con compuerta y luego la conexión de salto
        x = x_v * x_u + input
        return x

class Gated_FSMN_Block(nn.Module):
    """Bloque FSMN con compuerta, normalización y convoluciones."""
    def __init__(self,
                 dim,
                 inner_channels=256,
                 group_size=256, 
                 norm_type='scalenorm'):
        super(Gated_FSMN_Block, self).__init__()

        # se elige la normalización según 'norm_type'
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # primera convolución 1D para llegar a 'inner_channels'
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),  # convolución punto a punto
            nn.PReLU(),  # activación PReLU
        )

        # primera normalización (CLayerNorm, por canal)
        self.norm1 = CLayerNorm(inner_channels)

        # FSMN con compuerta para secuencias largas
        self.gated_fsmn = Gated_FSMN(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)

        # segunda normalización (por canal) después de FSMN
        self.norm2 = CLayerNorm(inner_channels)

        # segunda convolución 1D para regresar a 'dim'
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        """Pasada por el bloque Gated-FSMN."""
        
        # primera convolución y activación (se voltea para Conv1d)
        conv1 = self.conv1(input.transpose(2, 1))
        
        # se normaliza la salida de la primera convolución
        norm1 = self.norm1(conv1)
        
        # bloque FSMN con compuerta (se voltea para FSMN)
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))
        
        # se normaliza la salida de FSMN (se regresa para Conv1d)
        norm2 = self.norm2(seq_out.transpose(2, 1))
        
        # segunda convolución 1D
        conv2 = self.conv2(norm2)
        
        # se suma la entrada y regresa el resultado en el formato original
        return conv2.transpose(2, 1) + input

import torch.nn as nn

class Gated_FSMN_dilated(nn.Module):
    """FSMN con compuerta y convoluciones dilatadas."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lorder: int,
        hidden_size: int
    ):
        """Crea Gated_FSMN_dilated."""
        super().__init__()
        
        # convolución de la compuerta u
        self.to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        
        # convolución de la compuerta v
        self.to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        
        # se crea el FSMN dilatado
        self.fsmn = UniDeepFsmn_dilated(in_channels, out_channels, lorder, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pasada por el módulo FSMN con compuerta."""
        input = x  # se guarda la entrada para después
        x_u = self.to_u(x)  # pasa por la compuerta u
        x_v = self.to_v(x)  # pasa por la compuerta v
        
        x_u = self.fsmn(x_u)  # se aplica FSMN a la salida de la compuerta u
        
        # se juntan las compuertas u y v con la entrada
        x = x_v * x_u + input  # salida con compuerta y conexión residual
        
        return x  # regresa el tensor final

import torch.nn as nn

class Gated_FSMN_Block_Dilated(nn.Module):
    """Bloque FSMN con compuerta y convoluciones dilatadas."""

    def __init__(self,
                 dim: int,
                 inner_channels: int = 256,
                 group_size: int = 256, 
                 norm_type: str = 'scalenorm',
                 ):
        """Crea Gated_FSMN_Block_Dilated."""
        super(Gated_FSMN_Block_Dilated, self).__init__()

        # clase de normalización según el tipo
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm  # usa ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm  # usa LayerNorm

        self.group_size = group_size

        # convolución inicial con PReLU
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        
        self.norm1 = CLayerNorm(inner_channels)  # primera normalización
        
        # bloque FSMN con compuerta y convoluciones dilatadas
        self.gated_fsmn = Gated_FSMN_dilated(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        
        self.norm2 = CLayerNorm(inner_channels)  # segunda normalización
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)  # convolución de salida

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Pasada por el bloque FSMN con compuerta."""
        # primera convolución y se acomodan las dimensiones
        conv1 = self.conv1(input.transpose(2, 1))
        norm1 = self.norm1(conv1)  # normalización después de la primera convolución
        
        # pasa por el FSMN con compuerta y se regresan las dimensiones
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))
        norm2 = self.norm2(seq_out.transpose(2, 1))  # normalización después del FSMN
        
        # segunda convolución y regresa con la conexión residual
        conv2 = self.conv2(norm2)  # convolución final
        return conv2.transpose(2, 1) + input  # regresa la salida con la conexión residual


class MossformerBlock_GFSMN(nn.Module):
    """Bloque Mossformer2: atención con compuerta más bloques FSMN."""
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        causal=False,
        attn_dropout=0.1,
        norm_type='scalenorm',
        shift_tokens=True
    ):
        super().__init__()

        # revisa que el tipo de normalización exista
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        # se elige la normalización según el tipo
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # posiciones rotatorias con dimensión máxima de 32
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))

        # un bloque FSMN con compuerta por cada capa
        self.fsmn = nn.ModuleList([Gated_FSMN_Block_Dilated(dim) for _ in range(depth)])

        # lista de capas de atención FLASH
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens
            ) for _ in range(depth)
        ])

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        """Arma una secuencia de bloques UniDeep FSMN."""
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(
        self,
        x,
        *,
        mask=None
    ):
        """Pasada por el bloque Mossformer."""
        ii = 0
        # se recorren las capas de atención y los bloques FSMN
        for flash in self.layers:
            x = flash(x, mask=mask)  # capa de atención FLASH
            x = self.fsmn[ii](x)     # su bloque FSMN con compuerta
            ii += 1  # siguiente bloque FSMN

        return x  # regresa la salida después de todas las capas

class MossformerBlock(nn.Module):
    """Bloque Mossformer: serie de capas de atención con compuerta."""
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        causal=False,
        attn_dropout=0.1,
        norm_type='scalenorm',
        shift_tokens=True
    ):
        super().__init__()

        # revisa que el tipo de normalización exista
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        # se elige la normalización según el tipo
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # posiciones rotatorias, con dimensión máxima de 32
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))
        
        # capas de atención FLASH según la profundidad
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens
            ) for _ in range(depth)
        ])

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        """Arma una secuencia de bloques UniDeep FSMN."""
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(
        self,
        x,
        *,
        mask=None
    ):
        """Pasada por el bloque Mossformer."""
        ii = 0
        # se aplican todas las capas de atención a la entrada
        for flash in self.layers:
            x = flash(x, mask=mask)  # capa de atención FLASH
            ii += 1  # siguiente capa

        return x  # regresa la salida después de todas las capas


# -*- coding: utf-8 -*-
"""
Separación de dos voces encimadas.

Cuando dos personas hablan al mismo tiempo, Whisper escucha una mezcla y suele escribir solo a una
(o revuelve las dos frases). Aquí se parte ese pedazo de audio en dos "pistas", una por voz, para
transcribir cada una por separado y ver de quién es con la huella de voz.

Solo se usa en los tramos donde ya se detectaron dos voces a la vez, porque es lento y en el resto
del audio no hace falta. Con un solo micrófono la separación nunca queda perfecta: la meta es no perder
lo que dijo la segunda persona.

Modelos (se descargan con herramientas/descargar_separadores.py):
  mossformer2   MossFormer2_SS_16K de ClearerVoice-Studio (Apache-2.0)
  sepformer     SepFormer de SpeechBrain entrenado con ruido y eco (WHAMR!, 16 kHz)
"""
import os
import types

import numpy as np

CARPETA_MODELOS = "modelos"
SR = 16000


def disponible(modelo="mossformer2"):
    """¿Están descargados los pesos del separador?"""
    carpeta = os.path.join(CARPETA_MODELOS, "mossformer2-ss-16k" if modelo == "mossformer2" else "sepformer-whamr16k")
    return any(os.path.exists(os.path.join(carpeta, f)) for f in
               ("mossformer2_ss_16k.pt", "last_best_checkpoint", "masknet.ckpt"))


class Separador:
    def __init__(self, modelo="mossformer2", dispositivo="cpu"):
        import torch
        self.torch = torch
        self.modelo = modelo
        self.dispositivo = dispositivo
        if modelo == "mossformer2":
            from mossformer2_ss.mossformer2 import MossFormer2_SS_16K
            args = types.SimpleNamespace(encoder_embedding_dim=512, mossformer_sequence_dim=512,
                                         num_mossformer_layer=24, encoder_kernel_size=16, num_spks=2)
            self.red = MossFormer2_SS_16K(args).model
            carpeta = os.path.join(CARPETA_MODELOS, "mossformer2-ss-16k")
            ligero = os.path.join(carpeta, "mossformer2_ss_16k.pt")   # solo los pesos, en media precisión (112 MB)
            if os.path.exists(ligero):
                pesos = torch.load(ligero, map_location="cpu")
            else:                                                       # el original de 670 MB (trae el optimizador)
                with open(os.path.join(carpeta, "last_best_checkpoint"), encoding="utf-8") as f:
                    archivo = f.readline().strip()
                datos = torch.load(os.path.join(carpeta, archivo), map_location="cpu")
                pesos = datos.get("model", datos)
            pesos = {k.replace("module.", ""): v for k, v in pesos.items()}
            faltan = self.red.load_state_dict(pesos, strict=False)
            if faltan.missing_keys:
                print("MossFormer2: faltaron pesos:", len(faltan.missing_keys))
            self.red.to(dispositivo).eval()
            self.max_seg = 9.5          # su codificación de posición alcanza para unos 10 s
        elif modelo == "sepformer":
            try:
                from speechbrain.inference.separation import SepformerSeparation
            except ImportError:
                from speechbrain.pretrained import SepformerSeparation
            carpeta = os.path.join(CARPETA_MODELOS, "sepformer-whamr16k")
            opciones = dict(source=carpeta, savedir=carpeta, run_opts={"device": dispositivo})
            try:
                from speechbrain.utils.fetching import LocalStrategy
                self.red = SepformerSeparation.from_hparams(**opciones, local_strategy=LocalStrategy.NO_LINK)
            except (ImportError, TypeError):
                self.red = SepformerSeparation.from_hparams(**opciones)
            self.max_seg = 30.0
        else:
            raise ValueError(f"Modelo de separación desconocido: {modelo}")

    def separar(self, x):
        """x: audio de 16 kHz (numpy). Regresa dos pistas del mismo largo, con el volumen de la mezcla."""
        torch = self.torch
        x = np.asarray(x, dtype=np.float32)
        largo = len(x)
        maximo = int(self.max_seg * SR)
        if largo > maximo:      # pedazos muy largos no deberían llegar aquí; se recorta por seguridad
            x = x[:maximo]
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
        with torch.inference_mode():
            t = torch.from_numpy(x).unsqueeze(0).to(self.dispositivo)
            if self.modelo == "mossformer2":
                salidas = [s[0].float().cpu().numpy() for s in self.red(t)]
            else:
                est = self.red.separate_batch(t)            # [1, T, 2]
                salidas = [est[0, :, k].float().cpu().numpy() for k in range(est.shape[-1])]
        # qué tan fuerte salió la pista más débil contra la otra; si casi no suena, es puro residuo
        volumen = [float(np.sqrt(np.mean(s ** 2))) for s in salidas]
        self.relacion = min(volumen) / (max(volumen) + 1e-9)
        pistas = []
        for s in salidas:
            s = s[:len(x)]
            s = s / (np.sqrt(np.mean(s ** 2)) + 1e-9) * rms
            pistas.append(np.pad(s, (0, largo - len(s))).astype(np.float32))
        return pistas

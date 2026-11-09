# -*- coding: utf-8 -*-
"""
Descarga el traductor de inglés a español (opus-mt-en-es) y lo deja en modelos/traductor/
convertido a CTranslate2, para que la app traduzca rápido y sin internet.

Se corre una sola vez en la computadora donde se arma el .exe:
    pip install transformers sentencepiece
    python preparar_traductor.py
"""
import os
import shutil

import ctranslate2.converters
from huggingface_hub import snapshot_download

MODELO = "Helsinki-NLP/opus-mt-en-es"
DESTINO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos", "traductor")

if __name__ == "__main__":
    print("Descargando", MODELO, "...")
    origen = snapshot_download(MODELO)
    if os.path.isdir(DESTINO):
        shutil.rmtree(DESTINO)
    print("Convirtiendo a CTranslate2 (int8)...")
    ctranslate2.converters.TransformersConverter(origen).convert(DESTINO, quantization="int8")
    for f in ("source.spm", "target.spm"):
        shutil.copy(os.path.join(origen, f), DESTINO)
    print("Listo:", DESTINO)

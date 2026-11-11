# -*- coding: utf-8 -*-
"""
Descarga los modelos para separar voces encimadas y los deja en modelos/ (se corre una sola vez).

    .venv\\Scripts\\python herramientas\\descargar_separadores.py

  modelos/mossformer2-ss-16k   MossFormer2 de ClearerVoice-Studio (16 kHz)
"""
import os

from huggingface_hub import snapshot_download

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # la carpeta del proyecto
MODELOS = [
    ("alibabasglab/MossFormer2_SS_16K", "mossformer2-ss-16k"),
]

if __name__ == "__main__":
    for repo, carpeta in MODELOS:
        destino = os.path.join(AQUI, "modelos", carpeta)
        print(f"Descargando {repo} ...")
        try:
            snapshot_download(repo, local_dir=destino)
            total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(destino) for f in fs)
            print(f"  listo: {destino} ({total / 1e6:.0f} MB)")
        except Exception as e:
            print(f"  no se pudo descargar {repo}: {e}")
    # el archivo original de MossFormer2 pesa 670 MB porque trae el estado del entrenamiento;
    # para la app basta con los pesos en media precisión (112 MB), que es lo que se copia al .exe
    carpeta = os.path.join(AQUI, "modelos", "mossformer2-ss-16k")
    original = os.path.join(carpeta, "last_best_checkpoint.pt")
    ligero = os.path.join(carpeta, "mossformer2_ss_16k.pt")
    if os.path.exists(original) and not os.path.exists(ligero):
        import torch
        datos = torch.load(original, map_location="cpu")
        pesos = {k.replace("module.", ""): (v.half() if v.dtype == torch.float32 else v)
                 for k, v in datos.get("model", datos).items()}
        torch.save(pesos, ligero)
        print(f"Pesos ligeros guardados: {ligero} ({os.path.getsize(ligero) / 1e6:.0f} MB)")
    print("\nTerminado.")

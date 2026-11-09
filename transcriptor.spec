# -*- mode: python ; coding: utf-8 -*-
# Receta de PyInstaller para Transcriptor.exe (se usa desde construir_exe.bat).
# Sale una carpeta dist\Transcriptor con el .exe y todo lo que necesita.
# NO incluye voces, galería ni grabaciones: cada quien enrola a su equipo desde cero.
import glob
import os
import site
import sys

from PyInstaller.utils.hooks import collect_all, copy_metadata

datas, binaries, ocultos = [], [], []

# paquetes que cargan piezas por nombre o traen archivos de datos (modelos, onnx, yaml, dll)
for paquete in ("speechbrain", "hyperpyyaml", "silero_vad", "faster_whisper", "ctranslate2", "tokenizers",
                "onnxruntime", "av", "sounddevice", "_sounddevice_data", "soundfile", "_soundfile_data",
                "torchaudio", "huggingface_hub", "customtkinter", "darkdetect", "rotary_embedding_torch", "einops"):
    try:
        d, b, h = collect_all(paquete)
        datas += d
        binaries += b
        ocultos += h
    except Exception as e:
        print(f"AVISO: no se pudo recolectar {paquete}: {e}")

# algunos paquetes revisan su propia versión al importarse
for paquete in ("speechbrain", "torch", "torchaudio", "tqdm", "huggingface_hub", "tokenizers", "faster_whisper",
                "ctranslate2", "silero_vad", "numpy", "scipy", "regex", "requests", "filelock", "packaging",
                "hyperpyyaml", "ruamel.yaml", "sentencepiece", "onnxruntime", "av", "soundfile", "sounddevice",
                "joblib", "fsspec", "safetensors"):
    try:
        datas += copy_metadata(paquete)
    except Exception:
        pass

# DLL de CUDA instaladas con pip (nvidia-cublas-cu12, nvidia-cudnn-cu12...) y carpetas *.libs de
# paquetes reparados con delvewheel (av.libs, numpy.libs, scipy.libs...)
for base in site.getsitepackages() + [site.getusersitepackages()]:
    for dll in glob.glob(os.path.join(base, "nvidia", "*", "bin", "*.dll")):
        binaries.append((dll, os.path.relpath(os.path.dirname(dll), base)))
    for carpeta in glob.glob(os.path.join(base, "*.libs")):
        for dll in glob.glob(os.path.join(carpeta, "*.dll")):
            binaries.append((dll, os.path.basename(carpeta)))

datas += [("icono.ico", "."), ("idiomas.json", ".")]

a = Analysis(
    ["app.py"],
    pathex=[SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=ocultos + ["transcriptor", "motor", "traductor", "diagnostico", "separador", "mossformer2_ss", "mossformer2_ss.mossformer2",
                             "mossformer2_ss.mossformer2_block", "mossformer2_ss.conv_module", "mossformer2_ss.fsmn",
                             "mossformer2_ss.layer_norm", "PIL.Image", "PIL.ImageDraw", "tkinter", "tkinter.ttk", "tkinter.filedialog",
                             "tkinter.messagebox", "scipy.signal", "multiprocessing"],
    excludes=["matplotlib", "IPython", "pytest", "notebook", "jupyter", "pandas"],
    # torch y speechbrain necesitan sus archivos .py además del bytecode (TorchScript e inspect)
    module_collection_mode={"torch": "pyz+py", "speechbrain": "pyz+py", "torchaudio": "pyz+py"},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Transcriptor", icon="icono.ico",
          console=False, upx=False, debug=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Transcriptor", upx=False)

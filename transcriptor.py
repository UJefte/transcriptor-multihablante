#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcriptor multihablante en español - versión 2
Enrolamiento de voces + separación de hablantes (ventanas solapadas + Viterbi)
+ transcripción casi en tiempo real + pasada final de alta precisión.

Comandos:
  python transcriptor.py dispositivos               Lista los micrófonos disponibles
  python transcriptor.py enrolar --nombre Uriel     Graba una muestra de voz y actualiza la galería
  python transcriptor.py galeria                    Recalcula la galería con todo lo que hay en voces/
  python transcriptor.py vivo                       Escucha el micrófono y transcribe casi al momento
  python transcriptor.py vivo --final               ...y al terminar genera la transcripción final mejorada
  python transcriptor.py refinar sesion.wav         Transcripción final de alta precisión de un audio grabado
  python transcriptor.py archivo prueba.wav         Procesa un audio con el mismo flujo que "vivo" (para evaluar)
"""
import argparse
import datetime
import glob
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from math import gcd

import numpy as np

VERSION = "2.0"
SR = 16000                 # frecuencia de muestreo de trabajo
BLOQUE = 512               # muestras por bloque para el detector de voz (32 ms)
VENTANA_ID = 2.0           # ventanas del enrolamiento y la calibración (s)
VENTANA_SEP = 1.0          # ventanas solapadas para separar hablantes (s)
PASO_SEP = 0.25            # cada cuánto se evalúa quién habla (s)
PASO_RAPIDO = 0.5          # paso cuando en vivo se acumulan frases en cola (5 veces menos cálculo)
PENAL_CAMBIO = 0.2         # costo de cambiar de hablante en el Viterbi
AJUSTE_UMBRAL = 0.2        # las ventanas en vivo son más cortas y ruidosas que las del enrolamiento
DESC_SIMILITUD = 0.3       # similitud para juntar a un desconocido con un grupo ya visto
DESC_MIN_SEG = 1.5         # duración mínima para crear un grupo de desconocido nuevo (s)
SOLAPE_MIN_SEG = 1.0       # empate mínimo entre dos voces para marcar posible solapamiento (s)
CARPETA_VOCES = "voces"
ARCHIVO_GALERIA = "galeria.npz"
ARCHIVO_VOCABULARIO = "vocabulario.txt"
CARPETA_SALIDA = "transcripciones"
EXT_AUDIO = (".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".webm")
PROMPT_BASE = "Conversación en español entre estudiantes universitarios."
# Frases que Whisper suele inventar cuando no hay voz clara
ALUCINACIONES = ("amara.org", "subtítulos realizados", "suscríbete", "gracias por ver el video")

TEXTO_ENROLAMIENTO = """
Lee este párrafo en voz alta, con tu tono normal y a la distancia a la que
hablarás en la conversación. Cuando termines, sigue hablando libremente de
tu día hasta que se acabe el tiempo. SOLO debe hablar la persona que se enrola.

  "El viernes por la mañana, Jorge buscó un café cerca de la facultad.
   Mientras esperaba, revisó sus apuntes de redes y pensó en la práctica
   de inteligencia artificial. Afuera llovía con fuerza y los autobuses
   llegaban tarde, así que decidió quedarse a leer un rato más. Al final,
   su amiga Ximena le pidió ayuda con un ejercicio de bases de datos."
"""


# ---------------------------------------------------------------- utilidades

def hhmmss(s):
    s = int(s)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def normalizar(v):
    return v / (np.linalg.norm(v) + 1e-9)


def nivelar_volumen(x, pico_minimo=0.1, pico_objetivo=0.3, max_ganancia=10.0):
    """Sube el volumen SOLO si la voz llega baja (voces lejanas o bocina), para que Whisper no la ignore.
    El audio con volumen normal no se toca: en las pruebas, amplificarlo empeoró la transcripción.
    Usa el pico (no el promedio) para que los silencios no disparen la ganancia y saturen la voz."""
    if len(x) == 0:
        return x.astype(np.float32)
    pico = float(np.percentile(np.abs(x), 99.5)) + 1e-9
    if pico >= pico_minimo:
        return x.astype(np.float32)
    g = float(min(pico_objetivo / pico, max_ganancia))
    return np.clip(x * g, -1, 1).astype(np.float32)


def remuestrear(x, sr):
    if sr == SR:
        return x.astype(np.float32)
    from scipy.signal import resample_poly
    g = gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def cargar_audio(ruta):
    """Lee cualquier audio y lo regresa en mono, 16 kHz, float32."""
    try:
        import soundfile as sf
        data, sr = sf.read(ruta, dtype="float32", always_2d=True)
        return remuestrear(data.mean(axis=1), sr)
    except Exception:
        pass
    try:
        crudo = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", ruta, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
            capture_output=True, check=True).stdout
        return np.frombuffer(crudo, dtype=np.float32).copy()
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise SystemExit(f"No pude leer '{ruta}'. Conviértelo a WAV o instala ffmpeg.")


def leer_vocabulario(nombres):
    """Nombres enrolados + palabras de vocabulario.txt, para que Whisper las escriba bien."""
    palabras = list(nombres)
    if os.path.exists(ARCHIVO_VOCABULARIO):
        with open(ARCHIVO_VOCABULARIO, encoding="utf-8") as f:
            palabras += [l.strip() for l in f if l.strip() and not l.startswith("#")]
    vistos, unicas = set(), []
    for p in palabras:
        if p.lower() not in vistos:
            vistos.add(p.lower())
            unicas.append(p)
    return " ".join(unicas) if unicas else None


class Decimador:
    """Pasa audio en vivo de 32/48 kHz a 16 kHz sin cortes entre bloques."""

    def __init__(self, sr):
        self.factor = sr // SR
        if self.factor > 1:
            from scipy.signal import firwin
            self.b = firwin(101, 7000, fs=sr)
            self.zi = np.zeros(len(self.b) - 1)
        self.fase = 0

    def __call__(self, x):
        if self.factor == 1:
            return x.astype(np.float32)
        from scipy.signal import lfilter
        y, self.zi = lfilter(self.b, 1.0, x, zi=self.zi)
        salida = y[self.fase::self.factor]
        self.fase = (self.fase - len(x)) % self.factor
        return salida.astype(np.float32)


def elegir_sr(dispositivo, opciones):
    import sounddevice as sd
    for sr in opciones:
        try:
            sd.check_input_settings(device=dispositivo, samplerate=sr, channels=1, dtype="float32")
            return sr
        except Exception:
            continue
    raise SystemExit("El micrófono no acepta 16, 32 ni 48 kHz. Elige otro con --dispositivo "
                     "(ver: python transcriptor.py dispositivos).")


# ---------------------------------------------------------------- modelos

def dispositivo_torch():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _preparar_cuda_windows():
    """En Windows, agrega las DLL de CUDA instaladas con pip (nvidia-cublas-cu12, nvidia-cudnn-cu12)."""
    if os.name != "nt":
        return
    for base in sys.path:
        for carpeta in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(carpeta)
                os.environ["PATH"] = carpeta + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass


def hay_gpu_para_whisper():
    try:
        _preparar_cuda_windows()
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class Identificador:
    """Huella de voz (embedding ECAPA de SpeechBrain)."""

    def __init__(self, dispositivo):
        import torch
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        opciones = dict(source="speechbrain/spkrec-ecapa-voxceleb", savedir=os.path.join("modelos", "ecapa"),
                        run_opts={"device": dispositivo})
        try:  # En Windows evita el error de enlaces simbólicos
            from speechbrain.utils.fetching import LocalStrategy
            self.modelo = EncoderClassifier.from_hparams(**opciones, local_strategy=LocalStrategy.COPY)
        except (ImportError, TypeError):
            self.modelo = EncoderClassifier.from_hparams(**opciones)
        self.torch = torch
        self.dispositivo = dispositivo

    def embedding(self, x):
        return self.embeddings_lote([x])[0]

    def embeddings_lote(self, trozos, tam_lote=64):
        """Varias ventanas del mismo largo en una sola pasada del modelo (mucho más rápido)."""
        salida = []
        with self.torch.no_grad():
            for i in range(0, len(trozos), tam_lote):
                lote = np.stack([np.ascontiguousarray(t, dtype=np.float32) for t in trozos[i:i + tam_lote]])
                e = self.modelo.encode_batch(self.torch.from_numpy(lote).to(self.dispositivo))
                salida.append(e.reshape(len(lote), -1).cpu().numpy())
        E = np.concatenate(salida)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def cargar_whisper(nombre, dispositivo, quiet=False):
    from faster_whisper import WhisperModel
    gpu = dispositivo == "cuda" or hay_gpu_para_whisper()
    if nombre is None:
        nombre = "large-v3-turbo" if gpu else "small"
    intentos = ([("cuda", "float16")] if gpu else []) + [("cpu", "int8")]
    for dev, tipo in intentos:
        try:
            if not quiet:
                print(f"Cargando Whisper '{nombre}' en {dev}... (la primera vez se descarga)")
            modelo = WhisperModel(nombre, device=dev, compute_type=tipo)
            list(modelo.transcribe(np.zeros(SR, dtype=np.float32), language="es")[0])  # calentamiento
            if not quiet:
                print(f"Whisper '{nombre}' listo en {dev} ({tipo}).")
            return modelo, dev
        except Exception as e:
            print(f"  No se pudo usar Whisper '{nombre}' en {dev}: {e}")
    raise SystemExit(f"No se pudo cargar Whisper '{nombre}'.")


class DetectorVoz:
    """Detector de actividad de voz en streaming (Silero VAD)."""

    def __init__(self, umbral=0.5, silencio_ms=500):
        import torch
        from silero_vad import load_silero_vad, VADIterator
        self.torch = torch
        self.it = VADIterator(load_silero_vad(), threshold=umbral, sampling_rate=SR,
                              min_silence_duration_ms=silencio_ms, speech_pad_ms=30)

    def evento(self, bloque):
        r = self.it(self.torch.from_numpy(bloque))
        if not r:
            return None
        return "inicio" if "start" in r else "fin"


# ---------------------------------------------------------------- galería

class Galeria:
    def __init__(self, nombres, centroides, umbral):
        self.nombres = list(nombres)
        self.centroides = np.asarray(centroides, dtype=np.float32)
        self.umbral = float(umbral)

    @classmethod
    def cargar(cls, ruta=ARCHIVO_GALERIA):
        if not os.path.exists(ruta):
            raise SystemExit("No hay galería. Primero enrola voces: python transcriptor.py enrolar --nombre TuNombre")
        d = np.load(ruta)
        return cls([str(n) for n in d["nombres"]], d["centroides"], float(d["umbral"]))

    def guardar(self, ruta=ARCHIVO_GALERIA):
        np.savez(ruta, nombres=np.array(self.nombres), centroides=self.centroides, umbral=self.umbral)

    def comparar(self, emb):
        return sorted(((float(emb @ c), n) for n, c in zip(self.nombres, self.centroides)), reverse=True)


def recolectar_clips(carpeta):
    """voces/Nombre/*.wav  o  voces/Nombre.wav"""
    personas = {}
    if not os.path.isdir(carpeta):
        return personas
    for entrada in sorted(os.listdir(carpeta)):
        ruta = os.path.join(carpeta, entrada)
        if os.path.isdir(ruta):
            clips = [os.path.join(ruta, f) for f in sorted(os.listdir(ruta)) if f.lower().endswith(EXT_AUDIO)]
            if clips:
                personas.setdefault(entrada, []).extend(clips)
        elif entrada.lower().endswith(EXT_AUDIO):
            personas.setdefault(os.path.splitext(entrada)[0], []).append(ruta)
    return personas


def trocear(x, largo, salto):
    n, h = int(largo * SR), int(salto * SR)
    if len(x) <= n:
        return [x]
    return [x[i:i + n] for i in range(0, len(x) - n + 1, h)]


def quitar_silencios(trozos):
    rms = np.array([np.sqrt(np.mean(t ** 2)) for t in trozos])
    ref = np.percentile(rms, 90)
    return [t for t, r in zip(trozos, rms) if r >= 0.25 * ref]


def calibrar(embs, nombres, centroides):
    """Sugiere un umbral comparando la similitud con la voz propia y con voces ajenas."""
    mismos, ajenos = [], []
    for i, n in enumerate(nombres):
        s = embs[n] @ centroides.T
        mismos.extend(s[:, i])
        if len(nombres) > 1:
            ajenos.extend(np.delete(s, i, axis=1).ravel())
    p10 = float(np.percentile(mismos, 10))
    print(f"  Similitud con su propia voz:  típica {np.median(mismos):.2f}, baja {p10:.2f}")
    if ajenos:
        p90 = float(np.percentile(ajenos, 90))
        print(f"  Similitud con voces ajenas:   típica {np.median(ajenos):.2f}, alta {p90:.2f}")
        umbral = float(np.clip((p10 + p90) / 2, 0.2, 0.7))
        if p90 >= p10:
            print("  AVISO: algunas voces se confunden entre sí; graben más audio de enrolamiento.")
        S = centroides @ centroides.T
        for i in range(len(nombres)):
            for j in range(i + 1, len(nombres)):
                if S[i, j] > 0.6:
                    print(f"  AVISO: {nombres[i]} y {nombres[j]} suenan muy parecido ({S[i, j]:.2f}).")
    else:
        umbral = float(np.clip(p10 - 0.15, 0.2, 0.5))
        print("  Solo hay una persona enrolada; enrolen a más para calibrar mejor.")
    return umbral


def construir_galeria(ident, carpeta=CARPETA_VOCES):
    personas = recolectar_clips(carpeta)
    if not personas:
        raise SystemExit(f"No hay audios en '{carpeta}/'.")
    print("Calculando huellas de voz:")
    embs = {}
    for nombre, clips in personas.items():
        trozos, dur = [], 0.0
        for c in clips:
            x = cargar_audio(c)
            dur += len(x) / SR
            trozos += trocear(x, VENTANA_ID, VENTANA_ID / 2)
        trozos = quitar_silencios(trozos)
        largo = max(len(t) for t in trozos)
        mismos = [t for t in trozos if len(t) == largo]
        embs[nombre] = ident.embeddings_lote(mismos)
        aviso = "   AVISO: menos de 20 s, graba más" if dur < 20 else ""
        print(f"  {nombre}: {len(clips)} audio(s), {dur:.0f} s, {len(mismos)} ventanas útiles{aviso}")
    nombres = list(embs)
    centroides = np.stack([normalizar(embs[n].mean(axis=0)) for n in nombres])
    umbral = calibrar(embs, nombres, centroides)
    print(f"  Umbral sugerido: {umbral:.2f}")
    return Galeria(nombres, centroides, umbral)


# ---------------------------------------------------------------- segmentación en streaming

class Segmentador:
    """Junta bloques de 32 ms en frases. Corta al detectar silencio o al llegar a max_seg."""

    def __init__(self, vad, max_seg=6.0, min_seg=0.4, preroll=0.3):
        self.vad = vad
        self.max_m, self.min_m = int(max_seg * SR), int(min_seg * SR)
        self.preroll = deque(maxlen=max(1, int(preroll * SR / BLOQUE)))
        self.hablando = False
        self.buf, self.inicio, self.n = [], 0, 0

    def _emitir(self, bloques, inicio):
        if not bloques:
            return None
        audio = np.concatenate(bloques)
        return (inicio / SR, audio) if len(audio) >= self.min_m else None

    def _punto_de_corte(self):
        # Cortar en el bloque más silencioso del último segundo para no partir palabras
        ult = min(len(self.buf) - 1, int(SR / BLOQUE))
        rms = [np.sqrt(np.mean(b ** 2)) for b in self.buf[-ult:]]
        return len(self.buf) - ult + int(np.argmin(rms))

    def alimentar(self, b):
        salida = []
        ev = self.vad.evento(b)
        if not self.hablando:
            self.preroll.append(b)
            if ev == "inicio":
                self.hablando = True
                self.buf = list(self.preroll)
                self.inicio = self.n + len(b) - sum(len(x) for x in self.buf)
                self.preroll.clear()
        else:
            self.buf.append(b)
            if ev == "fin":
                salida.append(self._emitir(self.buf, self.inicio))
                self.buf, self.hablando = [], False
            elif sum(len(x) for x in self.buf) >= self.max_m:
                corte = self._punto_de_corte()
                salida.append(self._emitir(self.buf[:corte], self.inicio))
                self.inicio += sum(len(x) for x in self.buf[:corte])
                self.buf = self.buf[corte:]
        self.n += len(b)
        return [s for s in salida if s]

    def terminar(self):
        s = self._emitir(self.buf, self.inicio) if self.hablando else None
        self.buf, self.hablando = [], False
        return [s] if s else []


# ---------------------------------------------------------------- separación de hablantes

def viterbi(E, penal, inicial=None):
    """Ruta de estados que maximiza la suma de puntajes menos `penal` por cada cambio de hablante."""
    F, K = E.shape
    V = E[0].copy()
    if inicial is not None:
        V = V - penal * 0.5 * (np.arange(K) != inicial)
    atras = np.zeros((F, K), dtype=int)
    for f in range(1, F):
        mejor = int(V.argmax())
        quedarse = V >= V[mejor] - penal
        atras[f] = np.where(quedarse, np.arange(K), mejor)
        V = np.where(quedarse, V, V[mejor] - penal) + E[f]
    ruta = [int(V.argmax())]
    for f in range(F - 1, 0, -1):
        ruta.append(int(atras[f][ruta[-1]]))
    return ruta[::-1]


class Separador:
    """Decide quién habla en cada momento de una frase.

    1. Huellas de voz en ventanas de 1 s cada 0.25 s (solapadas).
    2. Cada paso de 0.25 s promedia la similitud de las ventanas que lo cubren.
    3. Viterbi elige la secuencia de hablantes más probable con un costo por cambiar,
       así un ruido aislado no corta el turno y los fragmentos cortos heredan el hablante correcto.
    4. Los tramos que no se parecen a nadie se agrupan por voz: Desconocido 1, Desconocido 2...
    """

    def __init__(self, ident, galeria, umbral, margen=0.05, agrupar_desconocidos=True):
        self.ident, self.galeria = ident, galeria
        self.u = umbral - AJUSTE_UMBRAL
        self.margen = margen
        self.agrupar = agrupar_desconocidos
        self.ultimo = None
        self.grupos = []          # desconocidos vistos: {"c": huella, "n": tramos, "nombre": ...}

    def _ventanas(self, audio, paso):
        n, L, h = len(audio), int(VENTANA_SEP * SR), int(paso * SR)
        if n <= L:
            return [(0, n)], self.ident.embeddings_lote([audio])
        inicios = list(range(0, n - L + 1, h))
        if inicios[-1] + L < n:
            inicios.append(n - L)
        return [(a, a + L) for a in inicios], self.ident.embeddings_lote([audio[a:a + L] for a in inicios])

    def _grupo_desconocido(self, e, dur):
        sims = [float(e @ g["c"]) for g in self.grupos]
        if sims and (max(sims) >= DESC_SIMILITUD or (dur < DESC_MIN_SEG and max(sims) >= 0.15)):
            g = self.grupos[int(np.argmax(sims))]
            if dur >= DESC_MIN_SEG:
                g["c"] = normalizar(g["c"] * g["n"] + e)
                g["n"] += 1
            return g["nombre"]
        if dur >= DESC_MIN_SEG or not self.grupos:
            nombre = f"Desconocido {len(self.grupos) + 1}"
            self.grupos.append({"c": e, "n": 1, "nombre": nombre})
            return nombre
        return "Desconocido"

    def etiquetar(self, audio, t0, rapido=False):
        paso = PASO_RAPIDO if rapido else PASO_SEP
        nombres, C = self.galeria.nombres, self.galeria.centroides
        K, h = len(nombres), int(paso * SR)
        vent, Em = self._ventanas(audio, paso)
        Sim = Em @ C.T
        F = max(1, int(np.ceil(len(audio) / h)))
        acc, cnt = np.zeros((F, K)), np.zeros(F)
        for (a, b), s in zip(vent, Sim):
            fa, fb = a // h, max(a // h + 1, int(np.ceil(b / h)))
            acc[fa:fb] += s
            cnt[fa:fb] += 1
        P = acc / np.maximum(cnt, 1)[:, None]
        E = np.concatenate([P, np.full((F, 1), self.u)], axis=1)
        ruta = viterbi(E, PENAL_CAMBIO, self.ultimo)
        self.ultimo = ruta[-1]
        # posible solapamiento: las dos voces enroladas más parecidas superan el umbral y casi empatan
        # durante al menos 1 s (en un cambio de turno normal el empate dura menos)
        solape = np.zeros(F, dtype=bool)
        if K > 1:
            orden = np.sort(P, axis=1)
            empate = (orden[:, -2] >= self.u) & (orden[:, -1] - orden[:, -2] < self.margen)
            minimo, f = max(2, int(round(SOLAPE_MIN_SEG / paso))), 0
            while f < F:
                g = f
                while g < F and empate[g]:
                    g += 1
                if g - f >= minimo:
                    solape[f:g] = True
                f = max(g, f + 1)
        tramos, f = [], 0
        while f < F:
            k = ruta[f]
            g = f
            while g < F and ruta[g] == k:
                g += 1
            ini, fin = t0 + f * paso, t0 + min(g * paso, len(audio) / SR)
            if k < K:
                nombre, score = nombres[k], float(P[f:g, k].mean())
            else:
                score = float(P[f:g].max(axis=1).mean()) if K else 0.0
                nombre = "Desconocido"
                if self.agrupar:
                    a, b = f * h, g * h
                    sel = [i for i, (va, vb) in enumerate(vent) if va < b and vb > a]
                    nombre = self._grupo_desconocido(normalizar(Em[sel].mean(axis=0)), (g - f) * paso)
            tramos.append({"ini": ini, "fin": fin, "nombre": nombre, "score": score,
                           "solape": bool(solape[f:g].any())})
            f = g
        return tramos


def ventana_en(vs, t):
    """Tramo que contiene el instante t; si cae en un hueco entre tramos, el más cercano."""
    for v in vs:
        if v["ini"] <= t < v["fin"]:
            return v
    return min(vs, key=lambda v: min(abs(t - v["ini"]), abs(t - v["fin"])))


def agrupar(palabras, vs):
    """Reparte las palabras (inicio, fin, texto) según quién hablaba en cada momento."""
    grupos = []
    for ini, fin, txt in palabras:
        v = ventana_en(vs, (ini + fin) / 2)
        if grupos and grupos[-1]["nombre"] == v["nombre"]:
            g = grupos[-1]
            g["fin"] = fin
            g["texto"] += txt
            g["scores"].append(v["score"])
            g["solape"] = g["solape"] or v["solape"]
            g["palabras"].append((ini, fin, txt))
        else:
            grupos.append({"inicio": ini, "fin": fin, "nombre": v["nombre"], "texto": txt,
                           "scores": [v["score"]], "solape": v["solape"], "palabras": [(ini, fin, txt)]})
    for g in grupos:
        g["texto"] = g["texto"].strip()
        g["score"] = sum(g["scores"]) / len(g["scores"])
    return unir_fragmentos([g for g in grupos if g["texto"]])


FIN_ORACION = (".", "?", "!", "…")


def unir_fragmentos(grupos, max_palabras=2, max_hueco=0.3):
    """Una o dos palabras sueltas que continúan una oración (sin punto de por medio y con la
    siguiente palabra en minúscula, p. ej. 'Ni' + 'tú ni Naomi...') se quedan con el hablante
    del resto de la oración. Las respuestas cortas reales ('Sí.', '¿Qué?') tienen su propia
    puntuación y no se tocan."""
    def continua(a, b):
        return (not a["texto"].endswith(FIN_ORACION) and b["texto"][:1].islower()
                and b["inicio"] - a["fin"] < max_hueco)

    def unir(a, b, nombre):
        return {"inicio": a["inicio"], "fin": b["fin"], "nombre": nombre, "texto": a["texto"] + " " + b["texto"],
                "scores": a["scores"] + b["scores"], "solape": a["solape"] or b["solape"],
                "score": (a["score"] + b["score"]) / 2, "palabras": a["palabras"] + b["palabras"]}

    salida = []
    i = 0
    while i < len(grupos):
        g = grupos[i]
        corto = len(g["texto"].split()) <= max_palabras
        if corto and i + 1 < len(grupos) and continua(g, grupos[i + 1]) and len(grupos[i + 1]["texto"].split()) > max_palabras:
            salida.append(unir(g, grupos[i + 1], grupos[i + 1]["nombre"]))
            i += 2
            continue
        if corto and salida and continua(salida[-1], g) and len(salida[-1]["texto"].split()) > max_palabras:
            salida[-1] = unir(salida[-1], g, salida[-1]["nombre"])
            i += 1
            continue
        salida.append(g)
        i += 1
    return salida


# ---------------------------------------------------------------- transcripción

def filtrar_segmentos(segmentos):
    palabras = []
    for s in segmentos:
        texto = (s.text or "").strip().lower()
        if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
            continue
        if any(a in texto for a in ALUCINACIONES):
            continue
        palabras.extend((w.start, w.end, w.word) for w in (s.words or []))
    return palabras


class Procesador:
    def __init__(self, ident, galeria, asr, umbral, margen, beam, contexto, ruta_txt, detalle=False,
                 vocabulario=None, agrupar_desconocidos=True):
        self.asr, self.beam = asr, beam
        self.contexto, self.detalle = contexto, detalle
        self.vocabulario = vocabulario
        self.separador = Separador(ident, galeria, umbral, margen, agrupar_desconocidos) if galeria else None
        self.ultimo_texto = ""
        self.palabras = []      # (momento, hablante, palabra) de toda la sesión, para evaluar
        self.f = open(ruta_txt, "a", encoding="utf-8") if ruta_txt else None

    def transcribir(self, audio):
        prompt = PROMPT_BASE
        if self.contexto and self.ultimo_texto:
            prompt += " " + self.ultimo_texto[-200:]
        audio = np.concatenate([nivelar_volumen(audio), np.zeros(int(0.4 * SR), np.float32)])
        segmentos, _ = self.asr.transcribe(
            audio, language="es", beam_size=self.beam, word_timestamps=True,
            condition_on_previous_text=False, vad_filter=False, initial_prompt=prompt,
            hotwords=self.vocabulario)
        return filtrar_segmentos(segmentos)

    def procesar(self, t0, audio, rapido=False):
        vs = self.separador.etiquetar(audio, t0, rapido)
        palabras = [(t0 + a, t0 + b, w) for a, b, w in self.transcribir(audio)]
        lineas = []
        grupos = agrupar(palabras, vs)
        for g in grupos:
            marca = " [posible solapamiento con otro hablante]" if g["solape"] else ""
            linea = f"[{g['nombre']}] ({hhmmss(g['inicio'])} - {hhmmss(g['fin'])}): {g['texto']}{marca}"
            if self.f:
                self.f.write(linea + "\n")
                self.f.flush()
            if self.detalle:
                linea += f"   (similitud {g['score']:.2f})"
            lineas.append(linea)
            self.ultimo_texto = g["texto"]
        for g in grupos:
            self.palabras.extend(((a + b) / 2, g["nombre"], w) for a, b, w in g["palabras"])
        return lineas

    def cerrar(self):
        if self.f:
            self.f.close()


# ---------------------------------------------------------------- pasada final (alta precisión)

def refinar_audio(x, galeria, ident, asr, umbral, margen=0.05, beam=5, vocabulario=None, progreso=True):
    """Transcripción final: Whisper ve todo el audio con su contexto completo y la separación
    de hablantes se hace sobre cada tramo de voz, manteniendo los grupos de desconocidos."""
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    # tramos de voz separados por al menos 1 s de silencio: así el Viterbi ve los turnos cortos con contexto
    regiones = get_speech_timestamps(torch.from_numpy(x), load_silero_vad(), sampling_rate=SR,
                                     min_silence_duration_ms=1000, speech_pad_ms=100)
    sep = Separador(ident, galeria, umbral, margen)
    tramos = []
    nivelado = x.astype(np.float32).copy()
    for i, r in enumerate(regiones):
        a, b = r["start"], r["end"]
        nivelado[a:b] = nivelar_volumen(x[a:b])      # cada tramo de voz con su propio nivel
        if b - a < int(0.3 * SR):
            continue
        tramos += sep.etiquetar(x[a:b], a / SR)
        if progreso:
            print(f"\r  Separando voces... {100 * (i + 1) // max(len(regiones), 1)}%", end="", flush=True)
    if progreso:
        print("\r  Separando voces... listo.          ")
        print("  Transcribiendo con contexto completo...")
    segmentos, _ = asr.transcribe(nivelado, language="es", beam_size=beam,
                                  word_timestamps=True, vad_filter=True, initial_prompt=PROMPT_BASE,
                                  hotwords=vocabulario)
    palabras = filtrar_segmentos(segmentos)
    if not tramos or not palabras:
        return [], []
    grupos = agrupar(palabras, tramos)
    # juntar intervenciones seguidas de la misma persona si la pausa es corta (texto más legible)
    unidos = []
    for g in grupos:
        if unidos and unidos[-1]["nombre"] == g["nombre"] and g["inicio"] - unidos[-1]["fin"] < 1.0:
            u = unidos[-1]
            u["fin"], u["texto"] = g["fin"], u["texto"] + " " + g["texto"]
            u["solape"] = u["solape"] or g["solape"]
        else:
            unidos.append(dict(g))
    lineas = []
    for g in unidos:
        marca = " [posible solapamiento con otro hablante]" if g["solape"] else ""
        lineas.append(f"[{g['nombre']}] ({hhmmss(g['inicio'])} - {hhmmss(g['fin'])}): {g['texto']}{marca}")
    etiquetadas = [((a + b) / 2, g["nombre"], w, a, b) for g in grupos for a, b, w in g["palabras"]]
    return lineas, etiquetadas


# ---------------------------------------------------------------- comandos

def preparar(args, modelo=None):
    galeria = Galeria.cargar()
    dev = dispositivo_torch()
    print(f"Cargando modelos (huella de voz en {dev})...")
    ident = Identificador(dev)
    asr, dev_asr = cargar_whisper(modelo if modelo is not None else args.modelo, dev)
    beam = args.beam if args.beam else (5 if dev_asr == "cuda" else 1)
    umbral = args.umbral if args.umbral is not None else galeria.umbral
    vocab = leer_vocabulario(galeria.nombres)
    print(f"Personas enroladas: {', '.join(galeria.nombres)} | umbral {umbral:.2f} | beam {beam}")
    return galeria, ident, asr, beam, umbral, vocab, dev_asr


def cmd_dispositivos(args):
    import sounddevice as sd
    print(sd.query_devices())
    print(f"\nMicrófono predeterminado: {sd.default.device[0]}")


def grabar(segundos, dispositivo):
    import sounddevice as sd
    sr = elegir_sr(dispositivo, (16000, 48000, 44100, 32000))
    audio = sd.rec(int(segundos * sr), samplerate=sr, channels=1, dtype="float32", device=dispositivo)
    for restante in range(int(segundos), 0, -1):
        print(f"\r  Grabando... {restante:3d} s restantes", end="", flush=True)
        time.sleep(1)
    sd.wait()
    print("\r  Grabación terminada.                ")
    return remuestrear(audio[:, 0], sr)


def cmd_enrolar(args):
    import soundfile as sf
    nombre = args.nombre.strip()
    print(TEXTO_ENROLAMIENTO)
    input(f"{nombre}, presiona Enter para empezar a grabar {args.segundos} s...")
    audio = grabar(args.segundos, args.dispositivo)
    pico = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if pico < 0.05:
        print("AVISO: el volumen salió muy bajo; acércate al micrófono y repite.")
    if np.mean(np.abs(audio) > 0.98) > 0.001:
        print("AVISO: el audio se saturó; aléjate un poco del micrófono y repite.")
    carpeta = os.path.join(CARPETA_VOCES, nombre)
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav")
    sf.write(ruta, audio, SR)
    print(f"Guardado en {ruta}")
    if not args.sin_galeria:
        cmd_galeria(args)


def cmd_galeria(args):
    ident = Identificador(dispositivo_torch())
    galeria = construir_galeria(ident)
    galeria.guardar()
    print(f"Galería guardada en {ARCHIVO_GALERIA} con: {', '.join(galeria.nombres)}")


def rutas_sesion(prefijo):
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    base = os.path.join(CARPETA_SALIDA, f"{prefijo}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    return base + ".txt", base + ".wav"


def resumen_latencias(latencias):
    if latencias:
        lat = np.array(latencias)
        print(f"Latencia (fin de frase -> texto): promedio {lat.mean():.1f} s, "
              f"mediana {np.median(lat):.1f} s, máxima {lat.max():.1f} s, en {len(lat)} frases.")


def ejecutar_refinado(ruta_wav, args, galeria=None, ident=None):
    """Genera <sesion>_final.txt con el modelo de la pasada final."""
    if galeria is None:
        galeria = Galeria.cargar()
    if ident is None:
        ident = Identificador(dispositivo_torch())
    modelo_final = args.modelo_final
    if modelo_final is None:
        modelo_final = "large-v3-turbo" if hay_gpu_para_whisper() else "small"
    asr, dev = cargar_whisper(modelo_final, dispositivo_torch())
    umbral = args.umbral if args.umbral is not None else galeria.umbral
    x = cargar_audio(ruta_wav)
    t = time.time()
    print(f"Pasada final sobre {len(x) / SR:.0f} s de audio con Whisper '{modelo_final}'...")
    lineas, _ = refinar_audio(x, galeria, ident, asr, umbral, args.margen, beam=5,
                              vocabulario=leer_vocabulario(galeria.nombres))
    salida = os.path.splitext(ruta_wav)[0] + "_final.txt"
    with open(salida, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + ("\n" if lineas else ""))
    print("\n".join(lineas))
    print(f"\nTranscripción final guardada en {salida}  ({time.time() - t:.0f} s)")


def cmd_refinar(args):
    ejecutar_refinado(args.ruta, args)


def cmd_vivo(args):
    import sounddevice as sd
    import soundfile as sf
    galeria, ident, asr, beam, umbral, vocab, _ = preparar(args)
    ruta_txt, ruta_wav = rutas_sesion("sesion")
    proc = Procesador(ident, galeria, asr, umbral, args.margen, beam, not args.sin_contexto, ruta_txt,
                      args.detalle, vocab)
    seg = Segmentador(DetectorVoz(args.vad, args.silencio_ms), max_seg=args.max_seg)
    sr = elegir_sr(args.dispositivo, (16000, 48000, 32000))
    decimar = Decimador(sr)
    q_audio, q_proc = queue.Queue(), queue.Queue()
    latencias, avisos = [], []
    candado = threading.Lock()

    def mostrar(texto):
        with candado:
            sys.stdout.write("\r" + " " * 70 + "\r" + texto + "\n")
            sys.stdout.flush()

    def medidor(rms, hablando):
        db = 20 * np.log10(rms + 1e-9)
        n = int(np.clip((db + 65) / 45 * 20, 0, 20))
        estado = "escuchando frase..." if hablando else ("sin sonido" if db < -60 else "")
        with candado:
            sys.stdout.write(f"\r  Mic [{'#' * n}{'-' * (20 - n)}] {db:4.0f} dB  {estado:<20}")
            sys.stdout.flush()

    def al_recibir(indata, frames, tiempo, estado):
        if estado:
            avisos.append(str(estado))
        q_audio.put(indata[:, 0].copy())

    def trabajador():
        while True:
            item = q_proc.get()
            if item is None:
                break
            t0, audio, listo = item
            try:
                # si ya hay otra frase esperando, esta se procesa en modo rápido para no atrasarse
                rapido = args.rapido or q_proc.qsize() > 0
                for linea in proc.procesar(t0, audio, rapido):
                    mostrar(linea)
            except Exception as e:
                mostrar(f"  (error procesando un fragmento: {e})")
            latencias.append(time.time() - listo)
            if args.detalle:
                mostrar(f"     latencia {latencias[-1]:.1f} s, en cola: {q_proc.qsize()}")

    hilo = threading.Thread(target=trabajador, daemon=True)
    hilo.start()
    wav = sf.SoundFile(ruta_wav, "w", SR, 1, "PCM_16")
    pendiente = np.zeros(0, dtype=np.float32)
    with sd.InputStream(device=args.dispositivo, samplerate=sr, channels=1, dtype="float32",
                        blocksize=int(sr * 0.05), callback=al_recibir):
        print(f"\nTranscripción: {ruta_txt}")
        print("\n" + "=" * 60)
        print("   ESCUCHANDO: ya puedes hablar.   (Ctrl+C para terminar)")
        print("=" * 60 + "\n")
        ultimo_medidor, nivel_max, inicio, avisado = 0.0, 0.0, time.time(), False
        try:
            while True:
                try:
                    x = decimar(q_audio.get(timeout=0.5))
                except queue.Empty:
                    continue
                wav.write(x)
                rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
                nivel_max = max(nivel_max, rms)
                ahora = time.time()
                if ahora - ultimo_medidor > 0.15:
                    medidor(rms, seg.hablando)
                    ultimo_medidor = ahora
                if not avisado and ahora - inicio > 8 and nivel_max < 0.003:
                    mostrar("AVISO: casi no llega sonido. Revisa que el micrófono no esté silenciado "
                            "o elige otro con --dispositivo.")
                    avisado = True
                pendiente = np.concatenate([pendiente, x])
                while len(pendiente) >= BLOQUE:
                    bloque, pendiente = pendiente[:BLOQUE], pendiente[BLOQUE:]
                    for t0, audio in seg.alimentar(bloque):
                        q_proc.put((t0, audio, time.time()))
        except KeyboardInterrupt:
            pass
    for t0, audio in seg.terminar():
        q_proc.put((t0, audio, time.time()))
    print("\nTerminando lo pendiente...")
    q_proc.put(None)
    hilo.join()
    wav.close()
    proc.cerrar()
    if avisos:
        print(f"AVISO: el micrófono reportó {len(avisos)} problema(s) de buffer; si es seguido, usa un modelo más chico.")
    resumen_latencias(latencias)
    print(f"Audio de la sesión: {ruta_wav}")
    print(f"Transcripción en vivo: {ruta_txt}")
    if args.final:
        print()
        ejecutar_refinado(ruta_wav, args, galeria, ident)
    else:
        print(f"Para la versión final más precisa:  python transcriptor.py refinar {ruta_wav}")


def procesar_archivo(x, galeria, ident, asr, umbral, margen, beam, contexto, ruta_txt, detalle, vocab,
                     vad=0.5, silencio_ms=500, max_seg=6.0, mostrar=print):
    """Mismo flujo que 'vivo' sobre un audio ya grabado (bloque por bloque)."""
    x = np.concatenate([x, np.zeros((-len(x)) % BLOQUE, dtype=np.float32)])
    proc = Procesador(ident, galeria, asr, umbral, margen, beam, contexto, ruta_txt, detalle, vocab)
    seg = Segmentador(DetectorVoz(vad, silencio_ms), max_seg=max_seg)
    frases = []
    for i in range(0, len(x), BLOQUE):
        frases += seg.alimentar(x[i:i + BLOQUE])
    frases += seg.terminar()
    for t0, audio in frases:
        for linea in proc.procesar(t0, audio):
            mostrar(linea)
    proc.cerrar()
    return proc.palabras


def cmd_archivo(args):
    galeria, ident, asr, beam, umbral, vocab, _ = preparar(args)
    x = cargar_audio(args.ruta)
    nombre = os.path.splitext(os.path.basename(args.ruta))[0]
    ruta_txt = args.salida or rutas_sesion(nombre)[0]
    if args.salida and os.path.exists(ruta_txt):
        os.remove(ruta_txt)
    inicio = time.time()
    procesar_archivo(x, galeria, ident, asr, umbral, args.margen, beam, not args.sin_contexto, ruta_txt,
                     args.detalle, vocab, args.vad, args.silencio_ms, args.max_seg)
    dur, tard = len(x) / SR, time.time() - inicio
    print(f"\nAudio de {dur:.0f} s procesado en {tard:.0f} s ({tard / max(dur, 1e-9):.2f}x tiempo real).")
    print(f"Transcripción guardada en {ruta_txt}")


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument("--dispositivo", type=int, default=None, help="número de micrófono (ver 'dispositivos')")
    comun.add_argument("--modelo", default=None,
                       help="Whisper en vivo: tiny, base, small, medium, large-v3-turbo (por defecto small en CPU, "
                            "large-v3-turbo con GPU)")
    comun.add_argument("--modelo-final", default=None,
                       help="Whisper de la pasada final (por defecto large-v3-turbo con GPU, small en CPU)")
    comun.add_argument("--umbral", type=float, default=None, help="similitud mínima para reconocer a alguien")
    comun.add_argument("--margen", type=float, default=0.05, help="diferencia mínima entre las 2 voces más parecidas")
    comun.add_argument("--beam", type=int, default=None, help="beam de Whisper (más alto = más preciso y lento)")
    comun.add_argument("--max-seg", type=float, default=6.0, help="duración máxima de cada fragmento (s)")
    comun.add_argument("--silencio-ms", type=int, default=500, help="silencio que cierra una frase (ms)")
    comun.add_argument("--vad", type=float, default=0.5, help="sensibilidad del detector de voz (0-1)")
    comun.add_argument("--sin-contexto", action="store_true", help="no pasar el texto anterior a Whisper")
    comun.add_argument("--detalle", action="store_true", help="muestra similitud y latencia por frase")

    p = argparse.ArgumentParser(description=f"Transcriptor multihablante en español v{VERSION}")
    sub = p.add_subparsers(dest="comando", required=True)
    sub.add_parser("dispositivos", help="lista micrófonos").set_defaults(fn=cmd_dispositivos)
    e = sub.add_parser("enrolar", parents=[comun], help="graba una muestra de voz")
    e.add_argument("--nombre", required=True)
    e.add_argument("--segundos", type=int, default=40)
    e.add_argument("--sin-galeria", action="store_true", help="solo grabar, sin recalcular la galería")
    e.set_defaults(fn=cmd_enrolar)
    sub.add_parser("galeria", parents=[comun], help="recalcula la galería").set_defaults(fn=cmd_galeria)
    v = sub.add_parser("vivo", parents=[comun], help="transcribe desde el micrófono")
    v.add_argument("--final", action="store_true", help="al terminar, genera la transcripción final mejorada")
    v.add_argument("--rapido", action="store_true", help="separación de voces más ligera (para computadoras lentas)")
    v.set_defaults(fn=cmd_vivo)
    r = sub.add_parser("refinar", parents=[comun], help="transcripción final de alta precisión de un audio")
    r.add_argument("ruta")
    r.set_defaults(fn=cmd_refinar)
    a = sub.add_parser("archivo", parents=[comun], help="procesa un audio grabado con el flujo en vivo")
    a.add_argument("ruta")
    a.add_argument("--salida", default=None, help="ruta del TXT de salida")
    a.set_defaults(fn=cmd_archivo)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

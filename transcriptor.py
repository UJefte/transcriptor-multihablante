#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcriptor multihablante en español (Opción A)
Enrolamiento de voces + identificación de hablante + transcripción casi en tiempo real.

Comandos:
  python transcriptor.py dispositivos               Lista los micrófonos disponibles
  python transcriptor.py enrolar --nombre Uriel     Graba una muestra de voz y actualiza la galería
  python transcriptor.py galeria                    Recalcula la galería con todo lo que hay en voces/
  python transcriptor.py vivo                       Escucha el micrófono y transcribe casi al momento
  python transcriptor.py archivo prueba.wav         Procesa un audio grabado con el mismo flujo que "vivo"
"""
import argparse
import datetime
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from math import gcd

import numpy as np

SR = 16000                 # frecuencia de muestreo de trabajo
BLOQUE = 512               # muestras por bloque para el detector de voz (32 ms)
VENTANA_ID = 2.0           # segundos por ventana para identificar al hablante
CARPETA_VOCES = "voces"
ARCHIVO_GALERIA = "galeria.npz"
CARPETA_SALIDA = "transcripciones"
EXT_AUDIO = (".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aac", ".webm")
PROMPT_BASE = "Conversación en español entre estudiantes universitarios."
# Frases que Whisper suele inventar cuando no hay voz clara
ALUCINACIONES = ("amara.org", "subtítulos realizados", "suscríbete", "gracias por ver el video")

TEXTO_ENROLAMIENTO = """
Lee este párrafo en voz alta, con tu tono normal y a la distancia a la que
hablarás en la conversación. Cuando termines, sigue hablando libremente de
tu día hasta que se acabe el tiempo.

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
        with self.torch.no_grad():
            t = self.torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0).to(self.dispositivo)
            e = self.modelo.encode_batch(t).squeeze().cpu().numpy()
        return normalizar(e)


def cargar_whisper(nombre, dispositivo):
    from faster_whisper import WhisperModel
    if nombre is None:
        nombre = "medium" if dispositivo == "cuda" else "small"
    intentos = ([("cuda", "float16")] if dispositivo == "cuda" else []) + [("cpu", "int8")]
    for dev, tipo in intentos:
        try:
            modelo = WhisperModel(nombre, device=dev, compute_type=tipo)
            list(modelo.transcribe(np.zeros(SR, dtype=np.float32), language="es")[0])  # calentamiento
            print(f"Whisper '{nombre}' cargado en {dev} ({tipo}).")
            return modelo, dev
        except Exception as e:
            print(f"  No se pudo usar Whisper en {dev}: {e}")
    raise SystemExit("No se pudo cargar Whisper.")


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
        embs[nombre] = np.stack([ident.embedding(t) for t in trozos])
        aviso = "   AVISO: menos de 20 s, graba más" if dur < 20 else ""
        print(f"  {nombre}: {len(clips)} audio(s), {dur:.0f} s, {len(trozos)} ventanas útiles{aviso}")
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


# ---------------------------------------------------------------- identificación + transcripción

def suavizar(vs):
    """Cada ventana toma la etiqueta mayoritaria entre ella y sus vecinas."""
    originales = [v["nombre"] for v in vs]
    for i, v in enumerate(vs):
        conteo = Counter(originales[max(0, i - 1): i + 2])
        top = max(conteo.values())
        if conteo[originales[i]] != top:
            v["nombre"] = max(conteo, key=conteo.get)


def etiquetar_ventanas(audio, t0, identificar):
    paso, n, pos, vs = int(VENTANA_ID * SR), len(audio), 0, []
    while pos < n:
        fin = min(pos + paso, n)
        if n - fin < paso // 2:
            fin = n
        nombre, score, solape = identificar(audio[pos:fin])
        vs.append({"ini": t0 + pos / SR, "fin": t0 + fin / SR, "nombre": nombre, "score": score, "solape": solape})
        pos = fin
    suavizar(vs)
    return vs


def ventana_en(vs, t):
    for v in vs:
        if v["ini"] <= t < v["fin"]:
            return v
    return vs[-1]


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
        else:
            grupos.append({"inicio": ini, "fin": fin, "nombre": v["nombre"], "texto": txt,
                           "scores": [v["score"]], "solape": v["solape"]})
    for g in grupos:
        g["texto"] = g["texto"].strip()
        g["score"] = sum(g["scores"]) / len(g["scores"])
    return [g for g in grupos if g["texto"]]


class Procesador:
    def __init__(self, ident, galeria, asr, umbral, margen, beam, contexto, ruta_txt, detalle=False):
        self.ident, self.galeria, self.asr = ident, galeria, asr
        self.umbral, self.margen, self.beam = umbral, margen, beam
        self.contexto, self.detalle = contexto, detalle
        self.ultimo_texto = ""
        self.f = open(ruta_txt, "a", encoding="utf-8") if ruta_txt else None

    def identificar(self, trozo):
        scores = self.galeria.comparar(self.ident.embedding(trozo))
        mejor_score, mejor = scores[0]
        nombre = mejor if mejor_score >= self.umbral else "Desconocido"
        solape = (len(scores) > 1 and mejor_score >= self.umbral
                  and (mejor_score - scores[1][0]) < self.margen)
        return nombre, mejor_score, solape

    def transcribir(self, audio):
        prompt = PROMPT_BASE
        if self.contexto and self.ultimo_texto:
            prompt += " " + self.ultimo_texto[-200:]
        segmentos, _ = self.asr.transcribe(
            audio, language="es", beam_size=self.beam, word_timestamps=True,
            condition_on_previous_text=False, vad_filter=False, initial_prompt=prompt)
        palabras = []
        for s in segmentos:
            texto = (s.text or "").strip().lower()
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            if any(a in texto for a in ALUCINACIONES):
                continue
            palabras.extend((w.start, w.end, w.word) for w in (s.words or []))
        return palabras

    def procesar(self, t0, audio):
        vs = etiquetar_ventanas(audio, t0, self.identificar)
        palabras = [(t0 + a, t0 + b, w) for a, b, w in self.transcribir(audio)]
        lineas = []
        for g in agrupar(palabras, vs):
            marca = " [posible solapamiento con otro hablante]" if g["solape"] else ""
            linea = f"[{g['nombre']}] ({hhmmss(g['inicio'])} - {hhmmss(g['fin'])}): {g['texto']}{marca}"
            if self.f:
                self.f.write(linea + "\n")
                self.f.flush()
            if self.detalle:
                linea += f"   (similitud {g['score']:.2f})"
            lineas.append(linea)
            self.ultimo_texto = g["texto"]
        return lineas

    def cerrar(self):
        if self.f:
            self.f.close()


# ---------------------------------------------------------------- comandos

def preparar(args):
    galeria = Galeria.cargar()
    dev = dispositivo_torch()
    print(f"Cargando modelos (dispositivo: {dev})...")
    ident = Identificador(dev)
    asr, dev_asr = cargar_whisper(args.modelo, dev)
    beam = args.beam if args.beam else (5 if dev_asr == "cuda" else 1)
    umbral = args.umbral if args.umbral is not None else galeria.umbral
    print(f"Personas enroladas: {', '.join(galeria.nombres)} | umbral {umbral:.2f} | beam {beam}")
    return galeria, ident, asr, beam, umbral


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


def cmd_vivo(args):
    import sounddevice as sd
    import soundfile as sf
    galeria, ident, asr, beam, umbral = preparar(args)
    ruta_txt, ruta_wav = rutas_sesion("sesion")
    proc = Procesador(ident, galeria, asr, umbral, args.margen, beam, not args.sin_contexto, ruta_txt, args.detalle)
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
                for linea in proc.procesar(t0, audio):
                    mostrar(linea)
            except Exception as e:
                mostrar(f"  (error procesando un fragmento: {e})")
            latencias.append(time.time() - listo)
            if args.detalle:
                mostrar(f"     latencia {latencias[-1]:.1f} s, en cola: {q_proc.qsize()}")

    hilo = threading.Thread(target=trabajador, daemon=True)
    hilo.start()
    wav = None if args.no_guardar_audio else sf.SoundFile(ruta_wav, "w", SR, 1, "PCM_16")
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
                if wav:
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
    if wav:
        wav.close()
        print(f"Audio de la sesión: {ruta_wav}")
    proc.cerrar()
    if avisos:
        print(f"AVISO: el micrófono reportó {len(avisos)} problema(s) de buffer; si es seguido, usa un modelo más chico.")
    resumen_latencias(latencias)
    print(f"Transcripción guardada en {ruta_txt}")


def cmd_archivo(args):
    galeria, ident, asr, beam, umbral = preparar(args)
    x = cargar_audio(args.ruta)
    x = np.concatenate([x, np.zeros((-len(x)) % BLOQUE, dtype=np.float32)])
    nombre = os.path.splitext(os.path.basename(args.ruta))[0]
    ruta_txt = args.salida or rutas_sesion(nombre)[0]
    if args.salida and os.path.exists(ruta_txt):
        os.remove(ruta_txt)
    proc = Procesador(ident, galeria, asr, umbral, args.margen, beam, not args.sin_contexto, ruta_txt, args.detalle)
    seg = Segmentador(DetectorVoz(args.vad, args.silencio_ms), max_seg=args.max_seg)
    inicio = time.time()
    frases = []
    for i in range(0, len(x), BLOQUE):
        frases += seg.alimentar(x[i:i + BLOQUE])
        while frases:
            t0, audio = frases.pop(0)
            for linea in proc.procesar(t0, audio):
                print(linea)
    for t0, audio in seg.terminar():
        for linea in proc.procesar(t0, audio):
            print(linea)
    proc.cerrar()
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
    comun.add_argument("--modelo", default=None, help="tamaño de Whisper: tiny, base, small, medium, large-v3")
    comun.add_argument("--umbral", type=float, default=None, help="similitud mínima para reconocer a alguien")
    comun.add_argument("--margen", type=float, default=0.05, help="diferencia mínima entre las 2 voces más parecidas")
    comun.add_argument("--beam", type=int, default=None, help="beam de Whisper (más alto = más preciso y lento)")
    comun.add_argument("--max-seg", type=float, default=6.0, help="duración máxima de cada fragmento (s)")
    comun.add_argument("--silencio-ms", type=int, default=500, help="silencio que cierra una frase (ms)")
    comun.add_argument("--vad", type=float, default=0.5, help="sensibilidad del detector de voz (0-1)")
    comun.add_argument("--sin-contexto", action="store_true", help="no pasar el texto anterior a Whisper")
    comun.add_argument("--detalle", action="store_true", help="muestra similitud y latencia por frase")

    p = argparse.ArgumentParser(description="Transcriptor multihablante en español")
    sub = p.add_subparsers(dest="comando", required=True)
    sub.add_parser("dispositivos", help="lista micrófonos").set_defaults(fn=cmd_dispositivos)
    e = sub.add_parser("enrolar", parents=[comun], help="graba una muestra de voz")
    e.add_argument("--nombre", required=True)
    e.add_argument("--segundos", type=int, default=40)
    e.add_argument("--sin-galeria", action="store_true", help="solo grabar, sin recalcular la galería")
    e.set_defaults(fn=cmd_enrolar)
    sub.add_parser("galeria", parents=[comun], help="recalcula la galería").set_defaults(fn=cmd_galeria)
    v = sub.add_parser("vivo", parents=[comun], help="transcribe desde el micrófono")
    v.add_argument("--no-guardar-audio", action="store_true")
    v.set_defaults(fn=cmd_vivo)
    a = sub.add_parser("archivo", parents=[comun], help="procesa un audio grabado")
    a.add_argument("ruta")
    a.add_argument("--salida", default=None, help="ruta del TXT de salida")
    a.set_defaults(fn=cmd_archivo)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

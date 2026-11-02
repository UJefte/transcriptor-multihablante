# -*- coding: utf-8 -*-
"""
Motor de la app: une la interfaz gráfica con transcriptor.py sin imprimir nada en consola.

Dos implementaciones con la MISMA interfaz:
  Motor          el real (SpeechBrain + Whisper + Silero), el mismo de 'python transcriptor.py ...'
  MotorSimulado  sin modelos ni micrófono; sirve para trabajar en la interfaz en cualquier PC
                 (python app.py --simulado)

Todo lo que tarda (cargar modelos, grabar, transcribir) se llama desde un hilo aparte; los avisos
hacia la interfaz llegan por la función `al_evento(tipo, datos)`:
  "linea"  {"nombre", "inicio", "fin", "texto", "solape"}   una intervención nueva
  "nivel"  {"rms", "hablando"}                               medidor del micrófono
  "aviso"  "texto"                                           mensaje para mostrar
"""
import datetime
import os
import queue
import random
import shutil
import threading
import time

import numpy as np

import transcriptor as tr

SR = tr.SR

# ------------------------------------------------------------------ guion del enrolamiento
# Cada paso pide un tono distinto: la huella final cubre cómo suena la persona al preguntar,
# emocionarse, hablar bajo o fuerte, y no solo leyendo en tono plano.
PASOS_ENROLAMIENTO = [
    {"id": "normal", "titulo": "Lectura normal", "segundos": 22, "min_voz": 10,
     "instruccion": "Lee con tu voz de siempre, a la distancia a la que vas a hablar en la conversación.",
     "texto": "El viernes por la mañana, Jorge buscó un café cerca de la facultad. Mientras esperaba, "
              "revisó sus apuntes de redes y pensó en la práctica de inteligencia artificial. Afuera "
              "llovía con fuerza y los autobuses llegaban tarde, así que decidió quedarse a leer un rato más."},
    {"id": "pregunta", "titulo": "Preguntas", "segundos": 12, "min_voz": 5,
     "instruccion": "Lee con tono de pregunta, como si de verdad quisieras saber la respuesta.",
     "texto": "¿Ya terminaste la práctica de redes? ¿A qué hora es la clase de mañana? "
              "¿Quién se llevó mi cargador? ¿De verdad crees que nos dé tiempo?"},
    {"id": "emocion", "titulo": "Emoción", "segundos": 12, "min_voz": 5,
     "instruccion": "Lee con emoción, como si te acabaran de dar una muy buena noticia.",
     "texto": "¡No puede ser, pasamos el examen! ¡Qué bueno que viniste! "
              "¡Por fin terminamos el proyecto! ¡Te quedó increíble!"},
    {"id": "bajo", "titulo": "Voz baja", "segundos": 12, "min_voz": 4, "pico_minimo": 0.015,
     "instruccion": "Habla bajito, como si alguien durmiera en el cuarto de al lado. Sin susurrar.",
     "texto": "Oye, no hagas ruido, ya se durmió. Mañana te paso los apuntes, ¿va? "
              "Nos vemos temprano en la entrada."},
    {"id": "fuerte", "titulo": "Voz fuerte", "segundos": 12, "min_voz": 4,
     "instruccion": "Habla fuerte, como si llamaras a alguien al otro lado del salón. Sin gritar.",
     "texto": "¡Oye, Ximena! ¡Aquí estamos, en la mesa del fondo! "
              "¡Tráete las copias, que ya va a empezar la clase!"},
    {"id": "rapido", "titulo": "Rápido", "segundos": 12, "min_voz": 5,
     "instruccion": "Lee rápido, como cuando cuentas algo con prisa.",
     "texto": "Salí tarde de la casa, el metro venía lleno, me bajé en la estación equivocada, corrí "
              "dos cuadras y aun así llegué justo cuando el profe estaba pasando lista."},
    {"id": "lento", "titulo": "Lento y con números", "segundos": 14, "min_voz": 5,
     "instruccion": "Lee despacio y claro, como si le dictaras a alguien.",
     "texto": "La clase es el martes a las once y media, en el salón doscientos cuatro. "
              "Mi número empieza con cinco, cinco, tres, ocho."},
    {"id": "tecnico", "titulo": "Términos técnicos", "segundos": 12, "min_voz": 5,
     "instruccion": "Lee normal, como lo dirías en clase.",
     "texto": "Hice commit y push a la branch main, pero el deploy del backend falló "
              "por un bug en el endpoint del login."},
    {"id": "libre", "titulo": "Plática libre", "segundos": 30, "min_voz": 12, "libre": True,
     "instruccion": "Platica libremente, como si hablaras con un amigo. Cuando lleves al menos "
                    "15 segundos puedes presionar «Terminar».",
     "texto": "Ideas: qué hiciste hoy, qué comiste, una película o serie que te guste, "
              "o tus planes para el fin de semana."},
]

CONSISTENCIA_MINIMA = 0.45   # una toma que se parece menos que esto a las demás: probablemente otra voz o ruido
PARECIDO_ALTO = 0.60         # dos personas enroladas que se parecen más que esto pueden confundirse
MISMA_PERSONA = 0.75         # tan parecido que quizá es la misma persona enrolada con otro nombre


def nombre_valido(nombre):
    nombre = (nombre or "").strip()
    if not nombre:
        return None, "Escribe un nombre."
    if any(c in nombre for c in '\\/:*?"<>|'):
        return None, "El nombre no puede llevar \\ / : * ? \" < > |"
    if nombre.lower().startswith("desconocido"):
        return None, "Ese nombre está reservado para las voces no enroladas."
    if len(nombre) > 30:
        return None, "Usa un nombre más corto (máximo 30 letras)."
    return nombre, None


def _db(x):
    return float(20 * np.log10(x + 1e-9))


class _Base:
    """Lo que comparten el motor real y el simulado."""
    pasos = PASOS_ENROLAMIENTO

    def calificar(self, info, paso):
        """Convierte las medidas de una toma en avisos para la persona."""
        avisos, grave = [], False
        if info["pico"] < paso.get("pico_minimo", 0.04):
            avisos.append("Se oyó muy bajo: acércate al micrófono o sube su volumen.")
            grave = True
        if info["saturado"]:
            avisos.append("El audio se saturó: aléjate un poco del micrófono.")
            grave = True
        if info["voz_seg"] < paso["min_voz"]:
            avisos.append(f"Solo se detectaron {info['voz_seg']:.0f} s de voz; se necesitan {paso['min_voz']} s.")
            grave = True
        if info["snr_db"] is not None and info["snr_db"] < 12:
            avisos.append("Hay bastante ruido de fondo; si puedes, busca un lugar más callado.")
        info["avisos"] = avisos
        info["ok"] = not grave
        info["calidad"] = "mala" if grave else ("regular" if avisos else "buena")
        return info


# ====================================================================== motor real

class Motor(_Base):
    simulado = False

    def __init__(self):
        self.ident = self.asr = self.galeria = None
        self.dev_asr, self.beam, self.modelo = "cpu", 1, "small"
        self._vad = None
        self._sesion = None

    # ---------------------------------------------------------------- modelos
    def cargar(self, al_progreso=lambda texto: None):
        tr._preparar_cuda_windows()
        dev = tr.dispositivo_torch()
        al_progreso("Cargando la huella de voz (SpeechBrain)...")
        self.ident = tr.Identificador(dev)
        al_progreso("Cargando el detector de voz...")
        self._cargar_vad()
        gpu = tr.hay_gpu_para_whisper()
        self.modelo = "large-v3-turbo" if gpu else "small"
        al_progreso(f"Cargando Whisper '{self.modelo}' (la primera vez se descarga y tarda)...")
        try:
            self.asr, self.dev_asr = tr.cargar_whisper(self.modelo, dev, quiet=True)
        except SystemExit as e:
            raise RuntimeError(str(e))
        self.beam = 5 if self.dev_asr == "cuda" else 1
        self.recargar_galeria()
        if self.galeria is not None and self.galeria.umbrales is None:
            # galería de una versión anterior: se recalcula una vez para tener umbral por persona
            al_progreso("Actualizando la galería de voces (umbral por persona)...")
            self.recalcular_galeria()
        return f"Whisper '{self.modelo}' en {self.dev_asr.upper()}"

    def _cargar_vad(self):
        if self._vad is None:
            from silero_vad import load_silero_vad
            self._vad = load_silero_vad()
        return self._vad

    def recargar_galeria(self):
        try:
            self.galeria = tr.Galeria.cargar()
        except SystemExit:
            self.galeria = None
        return self.galeria

    # ---------------------------------------------------------------- micrófonos
    def microfonos(self):
        """[(indice, nombre)] de las entradas de audio, sin repetir el mismo micrófono en varios drivers."""
        import sounddevice as sd
        try:
            pred = sd.default.device[0]
            api = sd.query_devices(pred)["hostapi"] if pred is not None and pred >= 0 else None
        except Exception:
            pred, api = None, None
        lista = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and (api is None or d["hostapi"] == api):
                lista.append((i, d["name"]))
        return lista, pred

    def grabar_toma(self, segundos, dispositivo=None, al_nivel=None, detener=None):
        """Graba hasta `segundos` (o hasta que se active `detener`) y regresa el audio a 16 kHz."""
        import sounddevice as sd
        try:
            sr = tr.elegir_sr(dispositivo, (16000, 48000, 44100, 32000))
        except SystemExit as e:
            raise RuntimeError(str(e))
        trozos = []

        def cb(indata, frames, tiempo, estado):
            x = indata[:, 0].copy()
            trozos.append(x)
            if al_nivel:
                al_nivel(float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0)

        inicio = time.time()
        with sd.InputStream(device=dispositivo, samplerate=sr, channels=1, dtype="float32",
                            blocksize=int(sr * 0.05), callback=cb):
            while time.time() - inicio < segundos and not (detener and detener.is_set()):
                time.sleep(0.05)
        audio = np.concatenate(trozos) if trozos else np.zeros(0, np.float32)
        return tr.remuestrear(audio, sr)

    def reproducir(self, audio):
        import sounddevice as sd
        sd.stop()
        sd.play(audio, SR)

    # ---------------------------------------------------------------- enrolamiento
    def analizar_toma(self, audio, paso):
        """Mide volumen, saturación, segundos de voz y ruido de fondo de una toma."""
        import torch
        from silero_vad import get_speech_timestamps
        x = np.ascontiguousarray(audio, dtype=np.float32)
        info = {"duracion": len(x) / SR, "pico": float(np.max(np.abs(x))) if len(x) else 0.0,
                "saturado": bool(len(x) and np.mean(np.abs(x) > 0.98) > 0.001),
                "voz_seg": 0.0, "voz_db": -120.0, "ruido_db": None, "snr_db": None}
        if len(x) >= SR // 2:
            reg = get_speech_timestamps(torch.from_numpy(x), self._cargar_vad(), sampling_rate=SR)
            mascara = np.zeros(len(x), bool)
            for r in reg:
                mascara[r["start"]:r["end"]] = True
            info["voz_seg"] = float(mascara.sum() / SR)
            if mascara.any():
                info["voz_db"] = _db(np.sqrt(np.mean(x[mascara] ** 2)))
            if (~mascara).sum() > 0.3 * SR:
                info["ruido_db"] = _db(np.sqrt(np.mean(x[~mascara] ** 2)))
                if mascara.any():
                    info["snr_db"] = info["voz_db"] - info["ruido_db"]
        return self.calificar(info, paso)

    def _huellas(self, audio):
        trozos = tr.trocear(audio, tr.VENTANA_ID, tr.VENTANA_ID / 2)
        trozos = [t for t in tr.quitar_silencios(trozos) if len(t) == int(tr.VENTANA_ID * SR)]
        if not trozos:
            return np.zeros((0, 192), np.float32)
        return self.ident.embeddings_lote(trozos)

    def revisar_tomas(self, nombre, tomas):
        """Compara las tomas entre sí y con las personas ya enroladas.
        tomas: {paso_id: audio}. Regresa un reporte para mostrar antes de guardar."""
        huellas = {pid: self._huellas(a) for pid, a in tomas.items()}
        validas = {pid: E for pid, E in huellas.items() if len(E)}
        if not validas:
            return {"consistencia": {}, "parecidos": [], "avisos": ["No se encontró voz suficiente en las tomas."],
                    "ok": False}
        consistencia = {}
        for pid, E in validas.items():
            otras = [F for q, F in validas.items() if q != pid]
            if otras:
                c = tr.centroide_robusto(np.concatenate(otras))
                consistencia[pid] = float((E @ c).mean())
        todas = np.concatenate(list(validas.values()))
        c = tr.centroide_robusto(todas)
        parecidos = []
        if self.galeria:
            for n, cg in zip(self.galeria.nombres, self.galeria.centroides):
                if n.lower() != nombre.lower():
                    parecidos.append((float(c @ cg), n))
            parecidos.sort(reverse=True)
        avisos = []
        malas = [pid for pid, v in consistencia.items() if v < CONSISTENCIA_MINIMA]
        titulos = {p["id"]: p["titulo"] for p in self.pasos}
        if malas:
            avisos.append("Estas tomas no se parecen a las demás (¿habló otra persona o hubo ruido?): "
                          + ", ".join(titulos.get(m, m) for m in malas) + ". Repítelas.")
        if parecidos and parecidos[0][0] >= MISMA_PERSONA:
            avisos.append(f"Esta voz es casi igual a la de {parecidos[0][1]} ({parecidos[0][0]:.2f}). "
                          "¿Es la misma persona? Cada quien debe enrolarse con su propia voz.")
        elif parecidos and parecidos[0][0] >= PARECIDO_ALTO:
            avisos.append(f"Se parece bastante a {parecidos[0][1]} ({parecidos[0][0]:.2f}); "
                          "el sistema podría confundirlas.")
        return {"consistencia": consistencia, "parecidos": parecidos[:3], "avisos": avisos,
                "ok": not malas and not (parecidos and parecidos[0][0] >= MISMA_PERSONA),
                "ventanas": int(len(todas)), "segundos_voz": float(sum(len(a) for a in tomas.values()) / SR)}

    def guardar_enrolamiento(self, nombre, tomas, reemplazar=True):
        """Guarda las tomas en voces/<nombre>/ y recalcula la galería de todas las personas."""
        import soundfile as sf
        carpeta = os.path.join(tr.CARPETA_VOCES, nombre)
        if reemplazar and os.path.isdir(carpeta):
            shutil.rmtree(carpeta)
        os.makedirs(carpeta, exist_ok=True)
        sello = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for pid, audio in tomas.items():
            sf.write(os.path.join(carpeta, f"{sello}_{pid}.wav"), audio, SR)
        return self.recalcular_galeria()

    def recalcular_galeria(self):
        if not tr.recolectar_clips(tr.CARPETA_VOCES):
            if os.path.exists(tr.ARCHIVO_GALERIA):
                os.remove(tr.ARCHIVO_GALERIA)
            self.galeria = None
            return None
        try:
            galeria = tr.construir_galeria(self.ident)
        except SystemExit as e:
            raise RuntimeError(str(e))
        galeria.guardar()
        self.galeria = galeria
        return galeria

    def borrar_persona(self, nombre):
        carpeta = os.path.join(tr.CARPETA_VOCES, nombre)
        if os.path.isdir(carpeta):
            shutil.rmtree(carpeta)
        suelto = [f for f in os.listdir(tr.CARPETA_VOCES) if os.path.splitext(f)[0] == nombre] \
            if os.path.isdir(tr.CARPETA_VOCES) else []
        for f in suelto:
            os.remove(os.path.join(tr.CARPETA_VOCES, f))
        return self.recalcular_galeria()

    def personas(self):
        """[{nombre, tomas, segundos, en_galeria}] de lo que hay en voces/."""
        import soundfile as sf
        salida = []
        nombres = set(self.galeria.nombres) if self.galeria else set()
        for nombre, clips in tr.recolectar_clips(tr.CARPETA_VOCES).items():
            seg = 0.0
            for c in clips:
                try:
                    seg += sf.info(c).duration
                except Exception:
                    pass
            salida.append({"nombre": nombre, "tomas": len(clips), "segundos": seg, "en_galeria": nombre in nombres})
        return salida

    def identificar(self, audio):
        """Prueba rápida: ¿a quién se parece esta voz? -> ([(similitud, nombre)], {nombre: umbral})"""
        if not self.galeria:
            raise RuntimeError("Todavía no hay nadie enrolado.")
        E = self._huellas(audio)
        if not len(E):
            raise RuntimeError("No se detectó voz suficiente; habla al menos 3 segundos.")
        niveles = dict(zip(self.galeria.nombres, (float(v) for v in self.galeria.niveles())))
        return self.galeria.comparar(tr.normalizar(E.mean(axis=0))), niveles

    def umbrales(self):
        """{nombre: umbral} con el que se acepta a cada persona en la conversación."""
        if not self.galeria:
            return {}
        return dict(zip(self.galeria.nombres, (float(v) for v in self.galeria.niveles())))

    # ---------------------------------------------------------------- conversación en vivo
    def iniciar(self, dispositivo, al_evento, espanglish=True):
        """Empieza a escuchar. Las intervenciones llegan por al_evento('linea', {...})."""
        import sounddevice as sd
        import soundfile as sf
        if self._sesion:
            raise RuntimeError("Ya hay una conversación en curso.")
        tr.ESPANGLISH["activo"] = espanglish
        try:
            sr = tr.elegir_sr(dispositivo, (16000, 48000, 32000))
        except SystemExit as e:
            raise RuntimeError(str(e))
        nombres = self.galeria.nombres if self.galeria else []
        umbral = self.galeria.umbral if self.galeria else 0.5
        proc = tr.Procesador(self.ident, self.galeria, self.asr, umbral, 0.05, self.beam, True, None,
                             vocabulario=tr.leer_vocabulario(nombres))
        seg = tr.Segmentador(tr.DetectorVoz(0.5, 600), max_seg=8.0)
        q_audio = queue.Queue()

        def al_recibir(indata, frames, tiempo, estado):
            q_audio.put(indata[:, 0].copy())

        stream = sd.InputStream(device=dispositivo, samplerate=sr, channels=1, dtype="float32",
                                blocksize=int(sr * 0.05), callback=al_recibir)
        ruta_txt, ruta_wav = tr.rutas_sesion("sesion")
        s = {"proc": proc, "seg": seg, "q_audio": q_audio, "q_proc": queue.Queue(), "stream": stream,
             "parar": threading.Event(), "ruta_wav": ruta_wav, "ruta_txt": ruta_txt,
             "wav": sf.SoundFile(ruta_wav, "w", SR, 1, "PCM_16"), "txt": open(ruta_txt, "w", encoding="utf-8"),
             "al_evento": al_evento, "decimar": tr.Decimador(sr), "inicio": time.time()}
        s["hilo_proc"] = threading.Thread(target=self._trabajador, args=(s,), daemon=True)
        s["hilo_cap"] = threading.Thread(target=self._capturar, args=(s,), daemon=True)
        self._sesion = s
        s["hilo_proc"].start()
        s["stream"].start()
        s["hilo_cap"].start()
        return ruta_wav

    def _capturar(self, s):
        pendiente = np.zeros(0, np.float32)
        ultimo, nivel_max, avisado = 0.0, 0.0, False
        while not s["parar"].is_set() or not s["q_audio"].empty():
            try:
                x = s["decimar"](s["q_audio"].get(timeout=0.2))
            except queue.Empty:
                continue
            s["wav"].write(x)
            rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
            nivel_max = max(nivel_max, rms)
            ahora = time.time()
            if ahora - ultimo > 0.08:
                s["al_evento"]("nivel", {"rms": rms, "hablando": s["seg"].hablando})
                ultimo = ahora
            if not avisado and ahora - s["inicio"] > 8 and nivel_max < 0.003:
                s["al_evento"]("aviso", "Casi no llega sonido: revisa que el micrófono no esté silenciado "
                                        "o elige otro en la lista.")
                avisado = True
            pendiente = np.concatenate([pendiente, x])
            while len(pendiente) >= tr.BLOQUE:
                bloque, pendiente = pendiente[:tr.BLOQUE], pendiente[tr.BLOQUE:]
                for t0, audio in s["seg"].alimentar(bloque):
                    s["q_proc"].put((t0, audio))

    def _procesar_frase(self, s, t0, audio, rapido):
        proc = s["proc"]
        if proc.separador:
            vs = proc.separador.etiquetar(audio, t0, rapido)
        else:
            vs = [{"ini": t0, "fin": t0 + len(audio) / SR, "nombre": "Desconocido", "score": 0.0, "solape": False}]
        palabras = [(t0 + a, t0 + b, w) for a, b, w in proc.transcribir(audio)]
        salida = []
        for g in tr.agrupar(palabras, vs):
            texto = tr.corregir(g["texto"])
            proc.ultimo_texto = texto
            salida.append({"nombre": g["nombre"], "inicio": g["inicio"], "fin": g["fin"],
                           "texto": texto, "solape": g["solape"]})
        return salida

    def _trabajador(self, s):
        while True:
            item = s["q_proc"].get()
            if item is None:
                break
            t0, audio = item
            try:
                for linea in self._procesar_frase(s, t0, audio, rapido=s["q_proc"].qsize() > 0):
                    s["txt"].write(formatear(linea) + "\n")
                    s["txt"].flush()
                    s["al_evento"]("linea", linea)
            except Exception as e:
                s["al_evento"]("aviso", f"Error procesando un fragmento: {e}")

    def detener(self):
        """Deja de escuchar, termina lo pendiente y regresa la ruta del audio grabado."""
        s, self._sesion = self._sesion, None
        if not s:
            return None
        s["stream"].stop()
        s["stream"].close()
        s["parar"].set()
        s["hilo_cap"].join()
        for t0, audio in s["seg"].terminar():
            s["q_proc"].put((t0, audio))
        s["q_proc"].put(None)
        s["hilo_proc"].join()
        s["wav"].close()
        s["txt"].close()
        return s["ruta_wav"]

    def en_curso(self):
        return self._sesion is not None

    # ---------------------------------------------------------------- pasada final
    def pasada_final(self, ruta_audio, al_avance=None, espanglish=True):
        """Transcripción más precisa de un audio completo (la misma de 'transcriptor.py refinar')."""
        if not self.galeria:
            raise RuntimeError("Enrola al menos a una persona antes de transcribir.")
        tr.ESPANGLISH["activo"] = espanglish
        x = tr.cargar_audio(ruta_audio)
        unidos, _ = tr.refinar_grupos(x, self.galeria, self.ident, self.asr, self.galeria.umbral, 0.05,
                                      beam=5, vocabulario=tr.leer_vocabulario(self.galeria.nombres),
                                      progreso=False, al_avance=al_avance)
        lineas = [{"nombre": g["nombre"], "inicio": g["inicio"], "fin": g["fin"], "texto": g["texto"],
                   "solape": g["solape"]} for g in unidos]
        # la versión final también se guarda sola, junto a las de la sesión
        base = os.path.splitext(os.path.basename(ruta_audio))[0]
        os.makedirs(tr.CARPETA_SALIDA, exist_ok=True)
        try:
            with open(os.path.join(tr.CARPETA_SALIDA, base + "_final.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(formatear(l) for l in lineas) + ("\n" if lineas else ""))
        except OSError:
            pass
        return lineas


def formatear(linea):
    marca = " [posible solapamiento con otro hablante]" if linea.get("solape") else ""
    return f"[{linea['nombre']}] ({tr.hhmmss(linea['inicio'])} - {tr.hhmmss(linea['fin'])}): {linea['texto']}{marca}"


# ====================================================================== motor simulado

_GUION = [
    ("Uriel", "Bueno, ya está grabando. ¿Empezamos con la parte del backend?"),
    ("Naomi", "Sí, pero primero dime si ya quedó el deploy de ayer."),
    ("Uriel", "Quedó a medias, el endpoint del login seguía regresando error."),
    ("Desconocido 1", "Yo revisé los logs y era el token, ya expiraba muy rápido."),
    ("Naomi", "¿Y ya lo subiste a la branch main o sigue en tu pull request?"),
    ("Uriel", "Sigue en el pull request, falta que alguien lo revise."),
    ("Desconocido 1", "Yo lo reviso en la tarde y hacemos el merge."),
    ("Naomi", "Perfecto, entonces mañana en la daily vemos lo del frontend."),
]


class MotorSimulado(_Base):
    """Misma interfaz que Motor, sin modelos ni micrófono. Para diseñar la interfaz en cualquier PC."""
    simulado = True

    def __init__(self):
        self._personas = {"Uriel": 128.0, "Naomi": 121.0}
        self._sesion = None
        self.modelo, self.dev_asr = "simulado", "cpu"
        self.galeria = True

    def cargar(self, al_progreso=lambda texto: None):
        for paso in ("Cargando la huella de voz...", "Cargando el detector de voz...", "Cargando Whisper..."):
            al_progreso(paso)
            time.sleep(0.4)
        return "Motor simulado (sin modelos)"

    def microfonos(self):
        return [(0, "Micrófono simulado"), (1, "Otro micrófono simulado")], 0

    def grabar_toma(self, segundos, dispositivo=None, al_nivel=None, detener=None):
        inicio, partes = time.time(), []
        while time.time() - inicio < segundos and not (detener and detener.is_set()):
            nivel = abs(np.sin(time.time() * 3)) * 0.08 + 0.01
            partes.append((np.random.randn(800) * nivel).astype(np.float32))
            if al_nivel:
                al_nivel(nivel)
            time.sleep(0.05)
        return np.concatenate(partes) if partes else np.zeros(0, np.float32)

    def reproducir(self, audio):
        pass

    def analizar_toma(self, audio, paso):
        dur = len(audio) / SR
        voz = dur * random.uniform(0.6, 0.9)
        info = {"duracion": dur, "pico": 0.4, "saturado": False, "voz_seg": voz, "voz_db": -22.0,
                "ruido_db": -55.0, "snr_db": 33.0}
        return self.calificar(info, paso)

    def revisar_tomas(self, nombre, tomas):
        time.sleep(0.8)
        consistencia = {pid: random.uniform(0.62, 0.85) for pid in tomas}
        parecidos = sorted(((random.uniform(0.05, 0.3), n) for n in self._personas if n.lower() != nombre.lower()),
                           reverse=True)
        return {"consistencia": consistencia, "parecidos": parecidos, "avisos": [], "ok": True,
                "ventanas": 60, "segundos_voz": sum(len(a) for a in tomas.values()) / SR}

    def guardar_enrolamiento(self, nombre, tomas, reemplazar=True):
        time.sleep(1.0)
        self._personas[nombre] = sum(len(a) for a in tomas.values()) / SR
        return True

    def recalcular_galeria(self):
        return True

    def borrar_persona(self, nombre):
        self._personas.pop(nombre, None)
        return True

    def personas(self):
        return [{"nombre": n, "tomas": 9, "segundos": s, "en_galeria": True} for n, s in self._personas.items()]

    def identificar(self, audio):
        nombres = list(self._personas) or ["Nadie"]
        sims = sorted(((random.uniform(0.1, 0.35), n) for n in nombres), reverse=True)
        sims[0] = (random.uniform(0.6, 0.8), sims[0][1])
        return sorted(sims, reverse=True), self.umbrales()

    def umbrales(self):
        return {n: 0.30 + 0.03 * i for i, n in enumerate(self._personas)}

    def iniciar(self, dispositivo, al_evento, espanglish=True):
        if self._sesion:
            raise RuntimeError("Ya hay una conversación en curso.")
        parar = threading.Event()

        def correr():
            t = 0.0
            for i, (nombre, texto) in enumerate(_GUION):
                dur = len(texto.split()) * 0.35
                fin = time.time() + 1.2
                while time.time() < fin and not parar.is_set():
                    al_evento("nivel", {"rms": abs(np.sin(time.time() * 5)) * 0.06, "hablando": True})
                    time.sleep(0.08)
                if parar.is_set():
                    return
                al_evento("linea", {"nombre": nombre, "inicio": t, "fin": t + dur, "texto": texto,
                                    "solape": i == 6})
                t += dur + 0.4
            while not parar.is_set():
                al_evento("nivel", {"rms": 0.001, "hablando": False})
                time.sleep(0.1)

        self._sesion = (parar, threading.Thread(target=correr, daemon=True))
        self._sesion[1].start()
        return "simulado.wav"

    def detener(self):
        if not self._sesion:
            return None
        parar, hilo = self._sesion
        self._sesion = None
        parar.set()
        hilo.join()
        return "simulado.wav"

    def en_curso(self):
        return self._sesion is not None

    def pasada_final(self, ruta_audio, al_avance=None, espanglish=True):
        for i in range(20):
            if al_avance:
                al_avance("Separando voces" if i < 8 else "Transcribiendo", (i % 8) / 8 if i < 8 else (i - 8) / 12)
            time.sleep(0.1)
        t, salida = 0.0, []
        for nombre, texto in _GUION:
            dur = len(texto.split()) * 0.35
            salida.append({"nombre": nombre, "inicio": t, "fin": t + dur, "texto": texto, "solape": False})
            t += dur + 0.4
        return salida

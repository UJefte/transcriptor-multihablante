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
import csv
import datetime
import json
import os
import queue
import random
import shutil
import threading
import time

import numpy as np

import transcriptor as tr
import traductor
import separador as sep_voces

SR = tr.SR

# ------------------------------------------------------------------ guion del enrolamiento
# Los textos se escribieron para que, entre todos, salgan todos los sonidos del español de México
# (las 5 vocales y los 17 consonánticos, incluidos ñ, ch, ll/y y la rr) y los grupos como pr, tr, bl,
# fl o gl, tal como se hace en los textos de lectura "equilibrados" de fonética (El viento norte y el
# sol, El ratón Arturo). La huella se compara en ventanas de 1 a 2 segundos, y en ventanas tan cortas
# ayuda que la persona ya haya dicho esos sonidos al enrolarse.
# Cada paso pide además un tono distinto (pregunta, emoción, bajo, fuerte, rápido, lento), porque la voz
# cambia mucho con el esfuerzo vocal y una huella grabada solo en tono normal reconoce peor esos casos.
# La plática libre al final captura cómo habla la persona sin leer.
PASOS_ENROLAMIENTO = [
    {"id": "normal", "titulo": "Lectura normal", "segundos": 30, "min_voz": 12,
     "instruccion": "Lee con tu voz de siempre, a la distancia a la que vas a hablar en la conversación.",
     "texto": 'El viernes temprano, Jimena llegó a la facultad con su chamarra gris y una mochila llena de planos y una regla. Quería imprimir el proyecto antes de la clase de redes, pero la fila frente al centro de cómputo era enorme. Un chavo de lentes le prestó su cargador y juntos revisaron las gráficas, el código y la bibliografía. Al final, el profesor aceptó la entrega y se fueron a desayunar chilaquiles bien picosos.'},
    {"id": "pregunta", "titulo": "Preguntas", "segundos": 12, "min_voz": 5,
     "instruccion": "Lee con tono de pregunta, como si de verdad quisieras saber la respuesta.",
     "texto": '¿Ya terminaste la práctica de redes? ¿A qué hora empieza el examen de mañana? ¿Quién se llevó mi cargador? ¿De verdad crees que nos dé tiempo de grabar todo?'},
    {"id": "emocion", "titulo": "Emoción", "segundos": 12, "min_voz": 5,
     "instruccion": "Aléjate un metro del micrófono y lee con emoción, como si te dieran una muy buena noticia.",
     "texto": '¡No puede ser, pasamos el examen! ¡Qué gusto que viniste! ¡Por fin terminamos el proyecto! ¡Te quedó increíble, muchas felicidades!'},
    {"id": "bajo", "titulo": "Voz baja", "segundos": 12, "min_voz": 4, "pico_minimo": 0.015,
     "instruccion": "Habla bajito, como si alguien durmiera en el cuarto de al lado. Sin susurrar.",
     "texto": 'Oye, no hagas ruido, el niño ya se durmió. Mañana te paso los apuntes, ¿va? Nos vemos temprano en la entrada del metro.'},
    {"id": "fuerte", "titulo": "Voz fuerte", "segundos": 12, "min_voz": 4,
     "instruccion": "Aléjate un metro del micrófono y habla fuerte, como si llamaras a alguien al otro lado del salón.",
     "texto": '¡Oye, Fernanda! ¡Aquí estamos, en la mesa del fondo! ¡Tráete las copias y el gis, que ya va a empezar la clase!'},
    {"id": "rapido", "titulo": "Rápido", "segundos": 12, "min_voz": 5,
     "instruccion": "Lee rápido, como cuando cuentas algo con prisa.",
     "texto": 'Salí tarde de la casa, el metro venía lleno, me bajé en la estación equivocada, corrí tres cuadras bajo la lluvia y aun así llegué justo cuando el profe estaba pasando lista.'},
    {"id": "lento", "titulo": "Lento y con números", "segundos": 16, "min_voz": 5,
     "instruccion": "Lee despacio y claro, como si le dictaras a alguien.",
     "texto": 'La clase es el martes a las once y media, en el salón doscientos cuatro. Mi número empieza con cinco, cinco, tres, ocho, y mi correo es jimena punto flores.'},
    {"id": "tecnico", "titulo": "Términos técnicos", "segundos": 14, "min_voz": 5,
     "instruccion": "Lee como lo dirías en clase, con todo y muletillas.",
     "texto": 'O sea, este, hice commit y push a la branch main, pero el deploy del backend falló por un bug en el endpoint del login. Bueno, mañana lo revisamos con el equipo, ¿no?'},
    {"id": "ingles", "titulo": "Frase en inglés", "segundos": 12, "min_voz": 4,
     "instruccion": "Lee en inglés como te salga, sin preocuparte por la pronunciación.",
     "texto": 'Good morning everyone, thank you for coming today. We finished the project last week and we are ready to show you how it works.'},
    {"id": "libre", "titulo": "Plática libre", "segundos": 30, "min_voz": 12, "libre": True,
     "instruccion": "Platica libremente, como si hablaras con un amigo. Cuando lleves al menos "
                    "15 segundos puedes detener la grabación.",
     "texto": "Ideas: qué hiciste hoy, qué comiste, una película o serie que te guste, "
              "o tus planes para el fin de semana. Habla como siempre, con tus muletillas."},
]

CONSISTENCIA_MINIMA = 0.45   # una toma que se parece menos que esto a las demás: probablemente otra voz o ruido
PARECIDO_ALTO = 0.60         # dos personas enroladas que se parecen más que esto pueden confundirse
MISMA_PERSONA = 0.75         # tan parecido que quizá es la misma persona enrolada con otro nombre


# ------------------------------------------------------------------ perfiles de entorno
# Cada lugar suena distinto: en un salón hay murmullo y eco, afuera hay ruido y las voces llegan
# más débiles. El perfil mueve cuatro perillas:
#   vad          qué tan seguro debe estar el detector para decir "aquí hay voz" (más alto = ignora murmullo)
#   silencio_ms  cuánto silencio cierra una frase en vivo
#   penal        cuánto le cuesta al Viterbi cambiar de persona (más alto = menos saltos falsos con ruido)
#   ajuste       cuánto se baja el umbral de cada persona (más alto = la reconoce aunque el ruido
#                ensucie su voz, pero también es más fácil confundir a alguien no registrado)
# "Cuarto silencioso" son los valores con los que se midió la versión 2.2.
PERFILES = {
    "silencioso": {"nombre": "Cuarto silencioso", "detalle": "Pocas personas, sin ruido",
                   "vad": 0.50, "silencio_ms": 600, "penal": 0.20, "ajuste": 0.12},
    "salon": {"nombre": "Salón de clase", "detalle": "Eco, murmullo de fondo",
              "vad": 0.60, "silencio_ms": 500, "penal": 0.25, "ajuste": 0.14},
    "exterior": {"nombre": "Exterior o ruidoso", "detalle": "Mucho ruido, voces lejanas",
                 "vad": 0.65, "silencio_ms": 450, "penal": 0.30, "ajuste": 0.16},
}
AJUSTES_INICIALES = {"perfil": "silencioso", "traducir": False, "espanglish": True, "pasada_final": True,
                     "microfono": None, "tema": "claro", "final_grande": True, "separar_voces": True,
                     "modo_final": None, "ausentes": [], "filtro_graves": True, "velocidades": {}}
ARCHIVO_AJUSTES = "ajustes.json"

# Qué pasa al terminar de grabar
MODOS = {
    "rapido": {"nombre": "Rápido", "detalle": "Solo lo que salió en vivo"},
    "mejorado": {"nombre": "Mejorado", "detalle": "Texto más preciso, sin separar voces"},
    "completo": {"nombre": "Completo", "detalle": "Además separa las voces encimadas"},
}
TOPE_SEPARAR = 0.30        # separar es lo más tardado: se revisa como máximo este pedazo del audio
TOPE_MINIMO_S = 20.0       # en grabaciones cortas se revisan al menos estos segundos
ARCHIVO_METRICAS = "metricas.csv"
ARCHIVO_METRICAS_PERSONAS = "metricas_personas.csv"
CANCELADO = "cancelado"


class Cancelado(Exception):
    pass


def tope_separar(duracion):
    return max(duracion * TOPE_SEPARAR, TOPE_MINIMO_S)


def leer_ajustes():
    ajustes = dict(AJUSTES_INICIALES)
    try:
        with open(ARCHIVO_AJUSTES, encoding="utf-8") as f:
            ajustes.update({k: v for k, v in json.load(f).items() if k in AJUSTES_INICIALES})
    except (OSError, ValueError):
        pass
    if ajustes["perfil"] not in PERFILES:
        ajustes["perfil"] = "silencioso"
    if ajustes["modo_final"] not in MODOS:     # se respeta lo que tenía elegido la versión anterior
        ajustes["modo_final"] = ("rapido" if not ajustes["pasada_final"] else
                                 "mejorado" if not ajustes["separar_voces"] else "completo")
    return ajustes


def guardar_ajustes(ajustes):
    try:
        with open(ARCHIVO_AJUSTES, "w", encoding="utf-8") as f:
            json.dump(ajustes, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def aplicar_perfil(clave):
    """Pone los valores del perfil en transcriptor.py (el Viterbi y los umbrales los leen de ahí)."""
    p = PERFILES.get(clave, PERFILES["silencioso"])
    tr.PENAL_CAMBIO = p["penal"]
    tr.AJUSTE_PERSONA = p["ajuste"]
    return p


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

    def __init__(self):
        self.ajustes = leer_ajustes()
        aplicar_perfil(self.ajustes["perfil"])

    def cambiar_ajuste(self, clave, valor):
        self.ajustes[clave] = valor
        guardar_ajustes(self.ajustes)
        if clave == "perfil":
            aplicar_perfil(valor)

    def perfil(self):
        return PERFILES[self.ajustes["perfil"]]

    # quién está en la plática: se guardan los que NO están, así alguien recién enrolado aparece activo
    def presente(self, nombre):
        return nombre not in self.ajustes.get("ausentes", [])

    def poner_presente(self, nombre, si):
        ausentes = [n for n in self.ajustes.get("ausentes", []) if n != nombre]
        if not si:
            ausentes.append(nombre)
        self.cambiar_ajuste("ausentes", ausentes)

    # tiempo estimado: se aprende de lo que ha tardado en este equipo
    def _medir(self, clave, veces):
        v = dict(self.ajustes.get("velocidades") or {})
        v[clave] = round(0.5 * v[clave] + 0.5 * veces, 3) if clave in v else round(veces, 3)
        self.cambiar_ajuste("velocidades", v)

    def estimar(self, duracion, modo):
        """Segundos que va a tardar, o None si este equipo todavía no ha medido nada."""
        if modo == "rapido":
            return 0.0
        v = self.ajustes.get("velocidades") or {}
        if "transcribir" not in v:
            return None
        t = duracion * v["transcribir"]
        if modo == "completo":
            if "separar" not in v:
                return None
            t += min(tope_separar(duracion), duracion) * v["separar"]
        return t

    def equipo(self):
        """Texto corto de con qué se está transcribiendo, para mostrarlo en la app."""
        donde = {"cuda": "tarjeta NVIDIA", "cpu": "procesador"}.get(self.dev_asr, self.dev_asr)
        tamano = "modelo grande" if "large" in self.modelo else ("modelo chico" if self.modelo == "small"
                                                                  else self.modelo)
        texto = f"{donde[:1].upper()}{donde[1:]} · {tamano}"
        if self.modelo == "small" and self.ajustes.get("final_grande", True):
            texto += " (versión final con el grande)"
        return texto

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
        super().__init__()
        self.ident = self.asr = self.galeria = None
        self.asr_grande = None
        self.separador = None
        self.dev_asr, self.beam, self.modelo = "cpu", 1, "small"
        self._vad = None
        self._sesion = None
        self._cancelar = threading.Event()

    # ---------------------------------------------------------------- modelos
    def cargar(self, al_progreso=lambda texto: None):
        tr._preparar_cuda_windows()
        dev = tr.dispositivo_torch()
        import torch
        torch.set_num_threads(max(2, (os.cpu_count() or 2) - 1))     # un núcleo libre para la ventana
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
        tr.DETECTAR_INGLES["asr"] = self.asr      # el modelo de en vivo también decide si una frase es inglés
        self.recargar_galeria()
        enrolados = set(tr.recolectar_clips(tr.CARPETA_VOCES))
        if self.galeria is not None and self.galeria.umbrales is None:
            # galería de una versión anterior: se recalcula una vez para tener umbral por persona
            al_progreso("Actualizando la galería de voces (umbral por persona)...")
            self.recalcular_galeria()
        elif enrolados and (self.galeria is None or set(self.galeria.nombres) != enrolados):
            # se copiaron voces a mano a la carpeta voces/ (por ejemplo, de otra computadora)
            al_progreso("Calculando las huellas de voz de las personas enroladas...")
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

    def grabar_toma(self, segundos, dispositivo=None, al_nivel=None, detener=None, al_saturar=None):
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
            if al_saturar and len(x) and np.mean(np.abs(x) >= 0.98) > 0.005:
                al_saturar()

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

    def parar_audio(self):
        import sounddevice as sd
        sd.stop()

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
            if self.ajustes.get("filtro_graves", True):
                audio = tr.filtro_graves(audio)
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

    def galeria_activa(self):
        """La galería solo con las personas que están en la plática."""
        g = self.galeria
        if not g:
            return None
        idx = [i for i, n in enumerate(g.nombres) if self.presente(n)]
        if not idx or len(idx) == len(g.nombres):
            return g
        return tr.Galeria([g.nombres[i] for i in idx], g.centroides[idx], g.umbral,
                          None if g.umbrales is None else g.umbrales[idx])

    def umbrales(self):
        """{nombre: umbral} con el que se acepta a cada persona en la conversación."""
        if not self.galeria:
            return {}
        return dict(zip(self.galeria.nombres, (float(v) for v in self.galeria.niveles())))

    def parecidos(self):
        """(nombres, matriz) de qué tanto se parecen las huellas de voz entre sí (1 = idénticas)."""
        if not self.galeria:
            return [], np.zeros((0, 0))
        C = self.galeria.centroides
        return list(self.galeria.nombres), np.clip(C @ C.T, -1, 1)

    # ---------------------------------------------------------------- traducción opcional del inglés
    def _vocabulario(self):
        # nombres de las personas + vocabulario.txt + términos técnicos: nada de eso se traduce
        nombres = list(self.galeria.nombres) if self.galeria else []
        return nombres + tr._lineas_utiles(tr.ARCHIVO_VOCABULARIO) + tr._lineas_utiles(tr.ARCHIVO_VOCAB_TECNICO)

    def preparar_traductor(self):
        """Carga el traductor si la opción está activa (se llama desde un hilo aparte)."""
        if self.ajustes.get("traducir"):
            traductor.cargar()

    def _traducir(self, texto):
        if not self.ajustes.get("traducir") or not texto:
            return texto
        try:
            return traductor.traducir_texto(texto, self._vocabulario())
        except Exception as e:      # si falla la traducción, mejor el texto original que nada
            print("No se pudo traducir:", e)
            return texto

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
        activa = self.galeria_activa()
        nombres = self.galeria.nombres if self.galeria else []
        umbral = self.galeria.umbral if self.galeria else 0.5
        proc = tr.Procesador(self.ident, activa, self.asr, umbral, 0.05, self.beam, True, None,
                             vocabulario=tr.leer_vocabulario(nombres))
        p = aplicar_perfil(self.ajustes["perfil"])
        seg = tr.Segmentador(tr.DetectorVoz(p["vad"], p["silencio_ms"]), max_seg=8.0)
        q_audio = queue.Queue()

        def al_recibir(indata, frames, tiempo, estado):
            q_audio.put(indata[:, 0].copy())

        stream = sd.InputStream(device=dispositivo, samplerate=sr, channels=1, dtype="float32",
                                blocksize=int(sr * 0.05), callback=al_recibir)
        ruta_txt, ruta_wav = tr.rutas_sesion("sesion")
        s = {"proc": proc, "seg": seg, "q_audio": q_audio, "q_proc": queue.Queue(), "stream": stream,
             "parar": threading.Event(), "ruta_wav": ruta_wav, "ruta_txt": ruta_txt,
             "wav": sf.SoundFile(ruta_wav, "w", SR, 1, "PCM_16"), "txt": open(ruta_txt, "w", encoding="utf-8"),
             "al_evento": al_evento, "decimar": tr.Decimador(sr), "inicio": time.time(), "atrasos": [], "lineas": [],
             "descartar": False, "graves": tr.FiltroGraves() if self.ajustes.get("filtro_graves", True) else None}
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
            pendiente = np.concatenate([pendiente, s["graves"](x) if s["graves"] else x])
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
            texto = self._traducir(texto)
            salida.append({"nombre": g["nombre"], "inicio": g["inicio"], "fin": g["fin"],
                           "texto": texto, "solape": g["solape"]})
        return salida

    def _trabajador(self, s):
        while True:
            item = s["q_proc"].get()
            if item is None:
                break
            if s["descartar"]:
                continue
            t0, audio = item
            try:
                for linea in self._procesar_frase(s, t0, audio, rapido=s["q_proc"].qsize() > 0):
                    s["txt"].write(formatear(linea) + "\n")
                    s["txt"].flush()
                    s["lineas"].append(linea)
                    s["al_evento"]("linea", linea)
                # cuántos segundos va atrás el texto respecto a lo que se está diciendo
                atraso = time.time() - s["inicio"] - (t0 + len(audio) / SR)
                s["atrasos"].append(max(atraso, 0.0))
                s["al_evento"]("atraso", atraso)
            except Exception as e:
                s["al_evento"]("aviso", f"Error procesando un fragmento: {e}")

    def detener(self, descartar=False):
        """Deja de escuchar, termina lo pendiente y regresa la ruta del audio grabado.
        Con descartar=True no se procesa lo pendiente y se borra la grabación."""
        s, self._sesion = self._sesion, None
        if not s:
            return None
        s["descartar"] = descartar
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
        if descartar:
            for ruta in (s["ruta_wav"], s["ruta_txt"]):
                try:
                    os.remove(ruta)
                except OSError as e:
                    print("No se pudo borrar:", e)
            return None
        import soundfile as sf
        self._ultima_sesion = {"ruta": s["ruta_wav"], "duracion": sf.info(s["ruta_wav"]).duration, "atrasos": s["atrasos"],
                               "lineas": s["lineas"]}
        return s["ruta_wav"]

    def en_curso(self):
        return self._sesion is not None

    # ---------------------------------------------------------------- pasada final
    def _asr_final(self, al_avance=None):
        """Modelo para la versión final. Sin tarjeta NVIDIA, en vivo se usa el chico (para que no se atrase),
        pero la versión final no tiene prisa: se carga el grande en el procesador. Tarda más (más o menos lo
        que dura el audio), pero el texto sale mucho mejor."""
        if self.dev_asr == "cuda" or self.modelo != "small" or not self.ajustes.get("final_grande", True):
            return self.asr
        if self.asr_grande is None:
            if al_avance:
                al_avance("Cargando el modelo grande", 0.0)
            try:
                self.asr_grande, _ = tr.cargar_whisper("large-v3-turbo", "cpu", quiet=True)
            except (SystemExit, Exception) as e:
                print("No se pudo cargar el modelo grande, se usa el chico:", e)
                return self.asr
        return self.asr_grande

    def _separador(self, siempre=False):
        """Separador de voces encimadas (MossFormer2). Se carga la primera vez que se necesita.
        Con siempre=True se carga aunque la opción de Ajustes esté apagada (pantalla Separar voces)."""
        if not sep_voces.disponible() or not (siempre or self.ajustes.get("separar_voces", True)):
            return None
        if self.separador is None:
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self.separador = sep_voces.Separador("mossformer2", dev)
            except Exception as e:
                print("No se pudo cargar el separador de voces:", e)
                self.ajustes["separar_voces"] = False      # para no intentarlo en cada transcripción
                return None
        return self.separador

    def cancelar(self):
        self._cancelar.set()

    def pasada_final(self, ruta_audio, al_avance=None, espanglish=True, modo=None):
        """Transcripción más precisa de un audio completo (la misma de 'transcriptor.py refinar')."""
        if not self.galeria:
            raise RuntimeError("Enrola al menos a una persona antes de transcribir.")
        modo = modo or self.ajustes["modo_final"]
        if modo == "rapido":          # si piden mejorar aunque esté en Rápido, se hace sin separar
            modo = "mejorado"
        self._cancelar.clear()
        tiempos, marca = {}, [None, time.time()]

        def avance(etapa, fraccion):
            if self._cancelar.is_set():
                raise Cancelado(CANCELADO)
            ahora = time.time()
            if marca[0]:
                tiempos[marca[0]] = tiempos.get(marca[0], 0.0) + ahora - marca[1]
            marca[0], marca[1] = etapa, ahora
            if al_avance:
                al_avance(etapa, fraccion)

        tr.ESPANGLISH["activo"] = espanglish
        aplicar_perfil(self.ajustes["perfil"])
        x = tr.cargar_audio(ruta_audio)
        if self.ajustes.get("filtro_graves", True):
            x = tr.filtro_graves(x)
        duracion = len(x) / SR
        asr = self._asr_final(avance)
        separador = self._separador(siempre=True) if modo == "completo" else None
        galeria = self.galeria_activa()
        stats = {}
        avance("Separando voces", 0.0)
        unidos, _ = tr.refinar_grupos(x, galeria, self.ident, asr, galeria.umbral, 0.05,
                                      beam=5, vocabulario=tr.leer_vocabulario(self.galeria.nombres),
                                      progreso=False, al_avance=avance, separador=separador, abierto=modo == "completo",
                                      tope_s=tope_separar(duracion), stats=stats)
        avance("Fin", 1.0)
        lineas = [{"nombre": g["nombre"], "inicio": g["inicio"], "fin": g["fin"],
                   "texto": self._traducir(g["texto"]), "solape": g["solape"]} for g in unidos]
        # la versión final también se guarda sola, junto a las de la sesión
        base = os.path.splitext(os.path.basename(ruta_audio))[0]
        os.makedirs(tr.CARPETA_SALIDA, exist_ok=True)
        try:
            with open(os.path.join(tr.CARPETA_SALIDA, base + "_final.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(formatear(l) for l in lineas) + ("\n" if lineas else ""))
        except OSError:
            pass
        t_quien, t_texto = tiempos.get("Separando voces", 0.0), tiempos.get("Transcribiendo", 0.0)
        t_enc = tiempos.get("Separando voces encimadas", 0.0)
        if duracion >= 3:
            self._medir("transcribir", (t_quien + t_texto) / duracion)
            if stats.get("s_revisados"):
                self._medir("separar", t_enc / stats["s_revisados"])
        niv = dict(zip(galeria.nombres, (float(v) for v in galeria.niveles())))
        dudoso = sum(g["fin"] - g["inicio"] for g in unidos
                     if g["nombre"] in niv and not g.get("separado") and g["score"] - niv[g["nombre"]] < 0.05)
        self._guardar_metricas(ruta_audio, duracion, modo, lineas, t_quien, t_texto, t_enc, stats, dudoso)
        return lineas

    def registrar_rapido(self, ruta):
        """Sin versión final: se guardan las métricas de lo que salió en vivo."""
        s = getattr(self, "_ultima_sesion", None)
        if s and s["ruta"] == ruta:
            self._guardar_metricas(ruta, s["duracion"], "rapido", s["lineas"], 0.0, 0.0, 0.0, {}, 0.0)

    def _guardar_metricas(self, ruta, duracion, modo, lineas, t_quien, t_texto, t_enc, stats, dudoso):
        habla = {}
        for l in lineas:
            nombre = "Desconocido" if l["nombre"].startswith("Desconocido") else l["nombre"]
            habla[nombre] = habla.get(nombre, 0.0) + l["fin"] - l["inicio"]
        total_habla = sum(habla.values()) or 1.0
        s = getattr(self, "_ultima_sesion", None)
        atrasos = s["atrasos"] if s and s["ruta"] == ruta else []
        trabajo = t_quien + t_texto + t_enc
        fila = {"fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "audio": os.path.basename(ruta),
                "duracion_s": round(duracion, 1), "modo": modo, "equipo": self.equipo(),
                "t_quien_habla_s": round(t_quien, 1), "t_transcribir_s": round(t_texto, 1),
                "t_separar_voces_s": round(t_enc, 1), "t_total_s": round(trabajo, 1),
                "veces_lo_que_dura": round(trabajo / max(duracion, 0.1), 2),
                "encimados_revisados": stats.get("candidatos", 0), "encimados_separados": stats.get("separados", 0),
                "s_revisados": round(stats.get("s_revisados", 0.0), 1), "s_separados": round(stats.get("s_separados", 0.0), 1),
                "pct_dudoso": round(float(100 * dudoso / total_habla), 1),
                "pct_desconocido": round(float(100 * habla.get("Desconocido", 0.0) / total_habla), 1),
                "atraso_vivo_prom_s": round(float(np.mean(atrasos)), 1) if atrasos else "",
                "atraso_vivo_max_s": round(float(np.max(atrasos)), 1) if atrasos else "",
                "habla_por_persona": "; ".join(f"{n} {v:.1f}s" for n, v in habla.items())}
        self.ultimas_metricas = fila
        os.makedirs(tr.CARPETA_SALIDA, exist_ok=True)
        ruta_csv = os.path.join(tr.CARPETA_SALIDA, ARCHIVO_METRICAS)
        try:
            nuevo = not os.path.exists(ruta_csv)
            with open(ruta_csv, "a", newline="", encoding="utf-8-sig" if nuevo else "utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(fila))
                if nuevo:
                    w.writeheader()
                w.writerow(fila)
        except OSError as e:     # por ejemplo, si el archivo está abierto en Excel
            print("No se pudieron guardar las métricas:", e)

    # ---------------------------------------------------------------- métricas por persona
    def metricas_personas(self, al_avance=lambda texto, avance: None):
        """Qué tan bien se reconoce a cada persona con sus propias tomas (cada toma contra las demás)."""
        g = self.galeria
        if not g:
            raise RuntimeError("Todavía no hay nadie enrolado.")
        clips = tr.recolectar_clips(tr.CARPETA_VOCES)
        niv = dict(zip(g.nombres, (float(v) for v in g.niveles())))

        def huellas(ruta, largo):
            x = tr.cargar_audio(ruta)
            trozos = [t for t in tr.quitar_silencios(tr.trocear(x, largo, largo / 2)) if len(t) == int(largo * SR)]
            return self.ident.embeddings_lote(trozos) if trozos else np.zeros((0, 192), np.float32)

        filas = []
        for i, nombre in enumerate(g.nombres):
            al_avance(f"Midiendo a {nombre}", i / len(g.nombres))
            propias = [r for r in clips.get(nombre, [])]
            otros = [c for n, c in zip(g.nombres, g.centroides) if n != nombre]
            fila = {"nombre": nombre, "tomas": len(propias), "umbral": round(niv[nombre], 2)}
            sims, aciertos = [], {}
            for largo in (1.0, 3.0):
                por_toma = [huellas(r, largo) for r in propias]
                bien, total = 0, 0
                for k, E in enumerate(por_toma):
                    resto = [F for j, F in enumerate(por_toma) if j != k and len(F)]
                    if not len(E) or not resto:
                        continue
                    c = tr.centroide_robusto(np.concatenate(resto))
                    s_propia = E @ c
                    if largo == 1.0:
                        sims.append(s_propia)
                    if otros:
                        s_otros = (E @ np.stack(otros).T).max(axis=1)
                        bien += int((s_propia > s_otros).sum())
                    else:
                        bien += len(E)
                    total += len(E)
                aciertos[largo] = 100 * bien / total if total else None
            if sims:
                todas = np.concatenate(sims)
                fila.update(bajo=round(float(np.percentile(todas, 10)), 2), tipico=round(float(np.median(todas)), 2),
                            alto=round(float(np.percentile(todas, 90)), 2))
            else:
                fila.update(bajo="", tipico="", alto="")
            if otros:
                k = int(np.argmax([float(g.centroides[g.nombres.index(nombre)] @ c) for c in otros]))
                demas = [n for n in g.nombres if n != nombre]
                fila["mas_parecido"] = demas[k]
                fila["parecido"] = round(float(g.centroides[g.nombres.index(nombre)] @ otros[k]), 2)
            else:
                fila["mas_parecido"], fila["parecido"] = "", ""
            fila["aciertos_1s"] = "" if aciertos[1.0] is None else round(aciertos[1.0], 1)
            fila["aciertos_3s"] = "" if aciertos[3.0] is None else round(aciertos[3.0], 1)
            filas.append(fila)
        al_avance("Listo", 1.0)
        os.makedirs(tr.CARPETA_SALIDA, exist_ok=True)
        try:
            with open(os.path.join(tr.CARPETA_SALIDA, ARCHIVO_METRICAS_PERSONAS), "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(filas[0]))
                w.writeheader()
                w.writerows(filas)
        except OSError as e:
            print("No se pudieron guardar las métricas por persona:", e)
        return filas

    def probar_microfono(self, dispositivo, al_nivel=None, detener=None):
        """10 segundos de prueba: dice si el micrófono se oye bien antes de grabar en serio."""
        audio = self.grabar_toma(10, dispositivo, al_nivel, detener)
        return audio, self.analizar_toma(audio, {"min_voz": 3})

    # ---------------------------------------------------------------- pantalla Separar voces
    def _texto_de(self, asr, audio):
        segmentos, _ = asr.transcribe(tr.nivelar_volumen(audio), language="es", beam_size=5, word_timestamps=True,
                                      vad_filter=True, condition_on_previous_text=False,
                                      initial_prompt=tr.prompt_base())
        texto = "".join(w for _, _, w in tr.filtrar_segmentos(segmentos)).strip()
        return tr.corregir(tr.quitar_repetidos(texto))

    def _duenos(self, pistas):
        """De quién es cada pista, sin repetir persona. Regresa [(nombre, parecido), (nombre, parecido)]."""
        anonimos = [("Desconocido 1", None), ("Desconocido 2", None)]
        galeria = self.galeria_activa()
        if not galeria:
            return anonimos
        niv = galeria.niveles()
        parecidos = []
        for p in pistas:
            trozos = [t for t in tr.quitar_silencios(tr.trocear(p, 1.0, 0.5)) if len(t) == SR]
            if trozos:
                parecidos.append((self.ident.embeddings_lote(trozos) @ galeria.centroides.T).mean(0))
            else:
                parecidos.append(np.zeros(len(galeria.nombres)) - 1)
        return elegir_duenos(parecidos, niv, galeria.nombres, anonimos)

    def separar_audio(self, x, al_avance=lambda texto, avance: None):
        """Separa un audio corto en dos pistas y dice de quién es cada una y qué dijo."""
        aplicar_perfil(self.ajustes["perfil"])
        t0 = time.time()
        al_avance("Cargando el separador de voces", 0.05)
        sep = self._separador(siempre=True)
        if sep is None:
            raise RuntimeError("No se encontró el modelo para separar voces (carpeta modelos/mossformer2-ss-16k).")
        al_avance("Preparando Whisper", 0.1)
        asr = self._asr_final()
        t1 = time.time()
        al_avance("Separando las dos voces (es la parte tardada)", 0.2)
        pistas = [np.clip(p, -1, 1) for p in sep.separar(x)]
        t2 = time.time()
        al_avance("Viendo de quién es cada voz", 0.7)
        duenos = self._duenos(pistas)
        al_avance("Transcribiendo la mezcla", 0.75)
        texto_mezcla = self._texto_de(asr, x)
        textos = []
        for i, p in enumerate(pistas):
            al_avance(f"Transcribiendo la pista {i + 1}", 0.82 + i * 0.09)
            textos.append(self._texto_de(asr, p))
        t3 = time.time()
        import torch
        return {"pistas": [{"audio": p, "nombre": d[0], "parecido": d[1], "texto": t}
                           for p, d, t in zip(pistas, duenos, textos)],
                "texto_mezcla": texto_mezcla, "cargar": t1 - t0, "separar": t2 - t1, "transcribir": t3 - t2,
                "equipo": "tarjeta NVIDIA" if torch.cuda.is_available() else "procesador"}


def elegir_duenos(parecidos, niveles, nombres, anonimos):
    """Cada pista se queda con la persona que más se le parece. Si las dos salen con la misma, se la queda
    la que se parece más y la otra busca a la siguiente. Si nadie pasa su umbral, queda como desconocida."""
    P = [np.asarray(p, dtype=np.float32) - niveles for p in parecidos]
    orden = sorted(range(2), key=lambda i: -float(P[i].max()))     # primero la pista más segura
    salida, usados = [None, None], set()
    for i in orden:
        libres = [k for k in np.argsort(-P[i]) if k not in usados]
        if libres and P[i][libres[0]] > 0:
            k = int(libres[0])
            usados.add(k)
            salida[i] = (nombres[k], float(parecidos[i][k]))
        else:
            salida[i] = anonimos[i]
    return salida


def formatear(linea):
    marca = " [posible solapamiento con otro hablante]" if linea.get("solape") else ""
    return f"[{linea['nombre']}] ({tr.hhmmss(linea['inicio'])} - {tr.hhmmss(linea['fin'])}): {linea['texto']}{marca}"


# ====================================================================== motor simulado

_GUION = [
    ("Uriel", "Oye, ¿ya viste si el viernes es la presentación de proyectos de inteligencia artificial?"),
    ("Alejandro", "Sí, el trece. Todavía falta hacer el deploy del backend y probar el ejecutable."),
    ("Uriel", "Ayer hice commit y push a la branch main, pero el deploy del backend falló."),
    ("Gerardo", "Yo me encargo del manual, pero necesito que me pasen las capturas."),
    ("Desconocido 1", "¿Ya empezaron?"),
    ("Alejandro", "Sí, ya estamos grabando. ¿Y ya lo subiste o sigue en tu pull request?"),
    ("Uriel", "Sigue en el pull request, falta que alguien lo revise."),
    ("Gerardo", "Yo lo reviso en la tarde y hacemos el merge."),
]


class MotorSimulado(_Base):
    """Misma interfaz que Motor, sin modelos ni micrófono. Para diseñar la interfaz en cualquier PC."""
    simulado = True

    def __init__(self):
        super().__init__()
        self._personas = {"Uriel": 128.0, "Alejandro": 121.0, "Gerardo": 96.0}
        self._sesion = None
        self.modelo, self.dev_asr = "large-v3-turbo", "cpu"
        self.galeria = True
        self._cancelar = threading.Event()

    def cargar(self, al_progreso=lambda texto: None):
        for paso in ("Cargando la huella de voz...", "Cargando el detector de voz...", "Cargando Whisper..."):
            al_progreso(paso)
            time.sleep(0.4)
        return "Motor simulado (sin modelos)"

    def microfonos(self):
        return [(0, "Micrófono simulado"), (1, "Otro micrófono simulado")], 0

    def grabar_toma(self, segundos, dispositivo=None, al_nivel=None, detener=None, al_saturar=None):
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

    def parar_audio(self):
        pass

    def separar_audio(self, x, al_avance=lambda texto, avance: None):
        for texto, avance in (("Cargando el separador de voces", 0.1), ("Separando las dos voces (es la parte tardada)", 0.4),
                              ("Transcribiendo la pista 1", 0.8), ("Transcribiendo la pista 2", 0.9)):
            al_avance(texto, avance)
            time.sleep(0.5)
        t = np.linspace(0, 1, len(x))
        pistas = [np.clip(x * (0.6 + 0.4 * np.sin(9 * t)), -1, 1), np.clip(x * (0.6 - 0.4 * np.sin(9 * t)), -1, 1)]
        return {"pistas": [{"audio": pistas[0], "nombre": "Isabel", "parecido": 0.41,
                            "texto": "El lunes hay junta de maestros a las ocho de la mañana."},
                           {"audio": pistas[1], "nombre": "Uriel", "parecido": 0.38,
                            "texto": "Yo tengo que entregar el proyecto el viernes trece."}],
                "texto_mezcla": "Yo tengo que entregar el proyecto el viernes.",
                "cargar": 0.5, "separar": 1.0, "transcribir": 1.0, "equipo": "simulado"}

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

    def parecidos(self):
        nombres = list(self._personas)
        rng = np.random.RandomState(len(nombres))
        M = rng.uniform(0.12, 0.3, (len(nombres), len(nombres)))
        M = (M + M.T) / 2
        np.fill_diagonal(M, 1.0)
        return nombres, M

    def preparar_traductor(self):
        time.sleep(0.3)

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

    def detener(self, descartar=False):
        if not self._sesion:
            return None
        parar, hilo = self._sesion
        self._sesion = None
        parar.set()
        hilo.join()
        return "simulado.wav"

    def en_curso(self):
        return self._sesion is not None

    def cancelar(self):
        self._cancelar.set()

    def registrar_rapido(self, ruta):
        pass

    def probar_microfono(self, dispositivo, al_nivel=None, detener=None):
        audio = self.grabar_toma(1.5, dispositivo, al_nivel, detener)
        return audio, self.analizar_toma(audio, {"min_voz": 0})

    def metricas_personas(self, al_avance=lambda texto, avance: None):
        time.sleep(0.5)
        return [{"nombre": n, "tomas": 9, "umbral": 0.20 + i / 50, "bajo": 0.38, "tipico": 0.52, "alto": 0.66,
                 "mas_parecido": "Uriel" if n != "Uriel" else "Gerardo", "parecido": 0.31,
                 "aciertos_1s": 92.4, "aciertos_3s": 98.5} for i, n in enumerate(self._personas)]

    def pasada_final(self, ruta_audio, al_avance=None, espanglish=True, modo=None):
        self._cancelar.clear()
        for i in range(20):
            if self._cancelar.is_set():
                raise Cancelado(CANCELADO)
            if al_avance:
                al_avance("Separando voces" if i < 8 else "Transcribiendo", (i % 8) / 8 if i < 8 else (i - 8) / 12)
            time.sleep(0.1)
        t, salida = 0.0, []
        for nombre, texto in _GUION:
            dur = len(texto.split()) * 0.35
            salida.append({"nombre": nombre, "inicio": t, "fin": t + dur, "texto": texto, "solape": False})
            t += dur + 0.4
        return salida

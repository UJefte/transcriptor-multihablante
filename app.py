#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcriptor multihablante v4 - la app de escritorio.

  python app.py              normal, con micrófono y modelos
  python app.py --simulado   sin modelos, para probar solo la interfaz

Pantallas: Conversación, Separar voces (para la demo), Personas, Ajustes y Enrolar.
"""
import os
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox

# Como .exe: el programa va junto a Transcriptor.exe, pero los datos de cada quien (voces,
# grabaciones, ajustes) van en %LOCALAPPDATA%. Así el .exe se pasa a otra compu sin llevarse voces.
CONGELADA = getattr(sys, "frozen", False)
AQUI = os.path.dirname(sys.executable) if CONGELADA else os.path.dirname(os.path.abspath(__file__))
INTERNO = getattr(sys, "_MEIPASS", AQUI)
if CONGELADA:
    DATOS = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "TranscriptorMultihablante")
    os.makedirs(DATOS, exist_ok=True)
    import shutil
    for _nombre in ("vocabulario.txt", "vocabulario_tecnico.txt", "correcciones.txt"):
        if not os.path.exists(os.path.join(DATOS, _nombre)) and os.path.exists(os.path.join(AQUI, _nombre)):
            shutil.copy(os.path.join(AQUI, _nombre), DATOS)     # copia editable para este usuario
    # sin consola: los mensajes y errores quedan en transcriptor.log para poder revisarlos
    try:
        _registro = open(os.path.join(DATOS, "transcriptor.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = _registro
        import faulthandler
        faulthandler.enable(_registro)
        print(f"\n===== inicio {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    except OSError:
        pass
else:
    DATOS = AQUI
os.chdir(DATOS)                      # voces/, galeria.npz, transcripciones/ y ajustes.json son relativos a esta carpeta
sys.path.insert(0, AQUI)
sys.path.insert(0, INTERNO)

import customtkinter as ctk   # noqa: E402
import numpy as np            # noqa: E402

import motor as mt            # noqa: E402
import traductor              # noqa: E402
import diagnostico            # noqa: E402

mt.tr.CARPETA_MODELOS = os.path.join(AQUI, "modelos")     # los modelos sí van junto al programa
traductor.CARPETA_MODELOS = mt.tr.CARPETA_MODELOS
mt.sep_voces.CARPETA_MODELOS = mt.tr.CARPETA_MODELOS

try:
    from PIL import Image, ImageDraw
except ImportError:          # sin Pillow la app funciona igual, solo sin íconos
    Image = ImageDraw = None

# ------------------------------------------------------------------ colores
C = {
    "fondo": "#F6F6F3", "tarjeta": "#FFFFFF", "borde": "#E7E6E1", "linea": "#EFEEEA",
    "texto": "#1B1B1A", "texto2": "#3D3C38", "suave": "#55544F", "tenue": "#6B6A64", "apagado": "#A3A29B",
    "acento": "#2563C9", "acento_hover": "#1D4E9E", "acento_suave": "#EEF3FC", "acento_borde": "#9DBBEA",
    "negro": "#1B1B1A", "negro_hover": "#3A3A38", "gris": "#F1F0EC", "gris_hover": "#E6E5E0",
    "rojo": "#D9483B", "rojo_hover": "#B93A2F", "verde": "#1E7A4F", "verde_suave": "#EAF5EE",
    "ambar": "#9A5B00", "ambar_suave": "#FBF1DF", "rojo_suave": "#FBE7E4", "barra": "#E2E1DC",
}
# (punto, fondo del círculo) de cada persona, en orden fijo
COLORES_PERSONA = [("#2A78D6", "#E3EEFB"), ("#E07B24", "#FBEBDD"), ("#149487", "#DDF2EF"),
                   ("#7C4DDB", "#ECE5FB"), ("#D6457A", "#FBE3EC"), ("#3E8E2F", "#E4F2E0"),
                   ("#B23A3A", "#F6E1E1"), ("#0E7BB8", "#DDEEF7")]
COLOR_DESCONOCIDO = ("#8A8983", "#EDEDEA")


def familia():
    disponibles = set(tkfont.families())
    for f in ("Segoe UI", "Inter", "SF Pro Text", "Helvetica Neue", "Ubuntu", "DejaVu Sans"):
        if f in disponibles:
            return f
    return "TkDefaultFont"


def mmss(seg):
    seg = int(max(seg, 0))
    return f"{seg // 60:02d}:{seg % 60:02d}"


# ------------------------------------------------------------------ íconos (dibujados con Pillow)
def _icono(nombre, color, tam=18):
    """Íconos de línea sencillos (como los de la propuesta). Se dibujan grandes y se reducen para que se vean finos."""
    if Image is None:
        return None
    k = 4
    s = tam * k
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    w = int(1.8 * k)

    def P(x, y):
        return (x / 24 * s, y / 24 * s)

    def linea(*pts):
        d.line([P(*p) for p in pts], fill=color, width=w, joint="curve")

    def circulo(cx, cy, r, relleno=False):
        a, b = P(cx - r, cy - r), P(cx + r, cy + r)
        if relleno:
            d.ellipse([a, b], fill=color)
        else:
            d.ellipse([a, b], outline=color, width=w)

    if nombre == "chat":
        d.rounded_rectangle([P(3, 4), P(21, 17)], radius=5 * k, outline=color, width=w)
        linea((8, 17), (6, 21), (12, 17))
    elif nombre == "personas":
        circulo(9, 8, 3.5)
        d.arc([P(2.5, 13.5), P(15.5, 26.5)], 180, 360, fill=color, width=w)
        circulo(17, 9, 2.5)
        d.arc([P(14, 14.5), P(22, 24)], 230, 360, fill=color, width=w)
    elif nombre == "ajustes":
        for y, x in ((6, 16), (12, 10), (18, 18)):
            linea((4, y), (20, y))
            circulo(x, y, 2.3, True)
    elif nombre == "mas":
        linea((12, 5), (12, 19))
        linea((5, 12), (19, 12))
    elif nombre == "check":
        linea((5, 12.5), (9.5, 17), (19, 7.5))
    elif nombre == "aviso":
        circulo(12, 12, 9)
        linea((12, 7.5), (12, 13))
        circulo(12, 16.5, 1.2, True)
    elif nombre == "atras":
        linea((15, 5), (8, 12), (15, 19))
    elif nombre == "onda":
        for x, h in ((4, 2), (8, 5), (12, 8), (16, 5), (20, 2)):
            linea((x, 12 - h), (x, 12 + h))
    elif nombre == "play":
        d.polygon([P(7, 4), P(20, 12), P(7, 20)], fill=color)
    elif nombre == "casa":
        linea((3, 11), (12, 4), (21, 11))
        linea((5, 10), (5, 20), (19, 20), (19, 10))
    elif nombre == "salon":
        d.rounded_rectangle([P(3, 4), P(21, 16)], radius=2 * k, outline=color, width=w)
        linea((8, 20), (16, 20))
        linea((12, 16), (12, 20))
    elif nombre == "sol":
        circulo(12, 12, 4)
        for a, b in (((12, 2), (12, 4.5)), ((12, 19.5), (12, 22)), ((2, 12), (4.5, 12)), ((19.5, 12), (22, 12)),
                     ((5, 5), (6.7, 6.7)), ((17.3, 17.3), (19, 19)), ((5, 19), (6.7, 17.3)), ((17.3, 6.7), (19, 5))):
            linea(a, b)
    elif nombre == "grabar":
        circulo(12, 12, 8, True)
    elif nombre == "parar":
        d.rounded_rectangle([P(5, 5), P(19, 19)], radius=3 * k, fill=color)
    elif nombre == "copiar":
        d.rounded_rectangle([P(8, 8), P(20, 20)], radius=2 * k, outline=color, width=w)
        linea((16, 8), (16, 4), (4, 4), (4, 16), (8, 16))
    elif nombre == "guardar":
        linea((12, 4), (12, 15))
        linea((7, 10), (12, 15), (17, 10))
        linea((4, 16), (4, 20), (20, 20), (20, 16))
    elif nombre == "basura":
        linea((4, 7), (20, 7))
        linea((9, 7), (9, 4), (15, 4), (15, 7))
        linea((6, 7), (7, 20), (17, 20), (18, 7))
    elif nombre == "refrescar":
        d.arc([P(4, 4), P(20, 20)], 300, 230, fill=color, width=w)
        linea((17, 2.5), (18, 7), (13.5, 7.5))
    elif nombre == "puntos":
        for x in (5, 12, 19):
            circulo(x, 12, 1.8, True)
    elif nombre == "archivo":
        linea((6, 3), (14, 3), (19, 8), (19, 21), (6, 21), (6, 3))
        linea((14, 3), (14, 8), (19, 8))
    return ctk.CTkImage(light_image=im, dark_image=im, size=(tam, tam))


# ------------------------------------------------------------------ piezas reutilizables
class Medidor(tk.Canvas):
    """Barritas de nivel del micrófono, como en la propuesta."""

    def __init__(self, padre, barras=12, alto=34, ancho_barra=4, separacion=3, fondo=None):
        ancho = barras * (ancho_barra + separacion)
        super().__init__(padre, width=ancho, height=alto, bg=fondo or C["tarjeta"], highlightthickness=0, bd=0)
        self.n, self.alto, self.ab, self.sep = barras, alto, ancho_barra, separacion
        self.niveles = [0.0] * barras
        self.ids = []
        for i in range(barras):
            x = i * (ancho_barra + separacion)
            self.ids.append(self.create_rectangle(x, alto / 2 - 2, x + ancho_barra, alto / 2 + 2,
                                                  fill=C["barra"], width=0))

    def reiniciar(self):
        self.niveles = [0.0] * self.n
        self._dibujar(False)

    def poner(self, rms, hablando=False):
        db = 20 * np.log10(rms + 1e-9)
        v = float(np.clip((db + 60) / 45, 0, 1))
        self.niveles = self.niveles[1:] + [v]
        self._dibujar(hablando or v > 0.25)

    def _dibujar(self, activo):
        for i, (rid, v) in enumerate(zip(self.ids, self.niveles)):
            h = max(4, v * self.alto)
            x = i * (self.ab + self.sep)
            self.coords(rid, x, (self.alto - h) / 2, x + self.ab, (self.alto + h) / 2)
            color = C["rojo"] if v >= 0.95 else (C["acento"] if activo and v > 0.12 else C["barra"])  # rojo: se va a saturar
            self.itemconfig(rid, fill=color)


_FORMAS = {}


def _forma(clave, ancho, alto, dibujar):
    """Dibuja una figura 4 veces más grande y deja que CTk la reduzca: así los bordes salen suaves."""
    if clave not in _FORMAS:
        k = 4
        im = Image.new("RGBA", (ancho * k, alto * k), (0, 0, 0, 0))
        dibujar(ImageDraw.Draw(im), k)
        _FORMAS[clave] = ctk.CTkImage(light_image=im, dark_image=im, size=(ancho, alto))
    return _FORMAS[clave]


def img_circulo(tam, color, figura=None):
    def dibujar(d, k):
        s = tam * k
        d.ellipse([0, 0, s - 1, s - 1], fill=color)
        m = s / 2
        if figura == "punto":
            r = s * 0.15
            d.ellipse([m - r, m - r, m + r, m + r], fill="#FFFFFF")
        elif figura == "cuadro":
            r = s * 0.14
            d.rounded_rectangle([m - r, m - r, m + r, m + r], radius=s * 0.04, fill="#FFFFFF")
    return _forma(("circulo", tam, color, figura), tam, tam, dibujar)


def img_interruptor(encendido, activo=True):
    fondo = (C["acento"] if activo else "#AFC6EE") if encendido else ("#C9C8C2" if activo else "#E2E1DC")

    def dibujar(d, k):
        w, h = 46 * k, 26 * k
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=h / 2, fill=fondo)
        m = 3 * k
        x = w - h + m if encendido else m
        d.ellipse([x, m, x + h - 2 * m, h - m], fill="#FFFFFF")
    return _forma(("interruptor", encendido, activo), 46, 26, dibujar)


class OpcionPerfil(ctk.CTkFrame):
    """Tarjeta para elegir el entorno: ícono, nombre y descripción, todo alineado a la izquierda."""

    def __init__(self, padre, titulo, detalle, icono, F, comando):
        super().__init__(padre, corner_radius=12, border_width=1, fg_color=C["tarjeta"], border_color=C["borde"],
                         cursor="hand2")
        self.comando, self.elegida, self.encima = comando, False, False
        partes = [ctk.CTkLabel(self, text="", image=icono, width=20, height=20, anchor="w"),
                  ctk.CTkLabel(self, text=titulo, font=F["negrita"], text_color=C["texto"], anchor="w", height=20),
                  ctk.CTkLabel(self, text=detalle, font=F["chico"], text_color=C["suave"], anchor="w",
                               justify="left", height=18)]
        partes[0].pack(anchor="w", padx=16, pady=(16, 8))
        partes[1].pack(anchor="w", fill="x", padx=16)
        partes[2].pack(anchor="w", fill="x", padx=16, pady=(2, 16))
        self.lbl_detalle = partes[2]
        ajustar_texto(partes[2], self, 34)
        for w in [self] + partes:
            w.bind("<Button-1>", lambda e: self.comando(), add="+")
            w.bind("<Enter>", lambda e: self._encima(True), add="+")
            w.bind("<Leave>", lambda e: self._encima(False), add="+")
            if w is not self:
                w.configure(cursor="hand2")

    def _encima(self, si):
        self.encima = si
        self.pintar(self.elegida)

    def pintar(self, elegida):
        self.elegida = elegida
        if elegida:
            fondo = "#EAF1FD" if self.encima else "#F4F8FE"
        else:
            fondo = C["gris"] if self.encima else C["tarjeta"]
        self.configure(fg_color=fondo, border_color=C["acento"] if elegida else C["borde"],
                       border_width=2 if elegida else 1)
        for w in self.winfo_children():
            w.configure(fg_color="transparent")


class BotonRedondo(ctk.CTkLabel):
    """Botón circular rojo de grabar/detener."""

    def __init__(self, padre, tam, comando):
        super().__init__(padre, text="", width=tam, height=tam, fg_color="transparent", cursor="hand2")
        self.tam, self.comando, self.activo, self.modo, self.encima = tam, comando, True, "grabar", False
        self.bind("<Enter>", lambda e: self._encima(True))
        self.bind("<Leave>", lambda e: self._encima(False))
        self.bind("<Button-1>", lambda e: self.activo and self.comando())
        self._dibujar()

    def _encima(self, si):
        self.encima = si
        self._dibujar()

    def _dibujar(self):
        if not self.activo:
            color = "#EDB3AD"
        else:
            color = C["rojo_hover"] if self.encima else C["rojo"]
        figura = "punto" if self.modo == "grabar" else "cuadro"
        self.configure(image=img_circulo(self.tam, color, figura), cursor="hand2" if self.activo else "arrow")

    def poner(self, modo=None, activo=None):
        if modo is not None:
            self.modo = modo
        if activo is not None:
            self.activo = bool(activo)
        self._dibujar()


class Interruptor(ctk.CTkLabel):
    """Interruptor de encendido/apagado con buen contraste sobre fondo blanco."""

    def __init__(self, padre, variable, comando=None):
        super().__init__(padre, text="", width=46, height=26, fg_color="transparent", cursor="hand2")
        self.var, self.comando, self.activo = variable, comando, True
        self.bind("<Button-1>", self._clic)
        variable.trace_add("write", lambda *a: self._dibujar())
        self._dibujar()

    def _clic(self, e=None):
        if not self.activo:
            return
        self.var.set(not self.var.get())
        if self.comando:
            self.comando()

    def _dibujar(self):
        self.configure(image=img_interruptor(bool(self.var.get()), self.activo))

    def habilitar(self, si):
        self.activo = bool(si)
        self.configure(cursor="hand2" if si else "arrow")
        self._dibujar()


def boton(padre, texto, comando, tipo="secundario", **kw):
    """Botones con los tres estilos de la propuesta: principal (negro), secundario (borde) y azul."""
    estilos = {
        "principal": dict(fg_color=C["negro"], hover_color=C["negro_hover"], text_color="#FFFFFF"),
        "azul": dict(fg_color=C["acento"], hover_color=C["acento_hover"], text_color="#FFFFFF"),
        "secundario": dict(fg_color=C["tarjeta"], hover_color=C["gris"], text_color=C["texto"],
                           border_width=1, border_color=C["borde"]),
        "suave": dict(fg_color=C["acento_suave"], hover_color="#E1EAFA", text_color=C["acento_hover"]),
        "peligro": dict(fg_color=C["tarjeta"], hover_color=C["rojo_suave"], text_color=C["rojo"],
                        border_width=1, border_color=C["borde"]),
    }
    opciones = dict(height=40, corner_radius=10, text_color_disabled=C["apagado"])
    opciones.update(estilos[tipo])
    opciones.update(kw)
    return ctk.CTkButton(padre, text=texto, command=comando, **opciones)


def ajustar_texto(etiqueta, contenedor, margen=40, maximo=None):
    """Parte el texto en renglones según el ancho real de la tarjeta.
    Con un ancho fijo se salía cuando Windows tiene la escala en 125 % o 150 %."""
    estado = {"pendiente": None, "ancho": None}

    def aplicar():
        estado["pendiente"] = None
        try:
            if not etiqueta.winfo_exists():
                return
            escala = ctk.ScalingTracker.get_widget_scaling(contenedor)
            ancho = int(contenedor.winfo_width() / escala - margen)
            if maximo:
                ancho = min(ancho, maximo)
            if ancho > 60 and ancho != estado["ancho"]:
                estado["ancho"] = ancho
                etiqueta.configure(wraplength=ancho)
        except tk.TclError:
            pass

    def cambio(e=None):
        # mientras se arrastra la orilla de la ventana llegan muchos eventos; solo se aplica el último
        try:
            if estado["pendiente"] is not None:
                etiqueta.after_cancel(estado["pendiente"])
            estado["pendiente"] = etiqueta.after(60, aplicar)
        except tk.TclError:
            pass
    contenedor.bind("<Configure>", cambio, add="+")
    etiqueta.after(30, aplicar)
    return etiqueta


def tarjeta(padre, **kw):
    opciones = dict(fg_color=C["tarjeta"], corner_radius=16, border_width=1, border_color=C["borde"])
    opciones.update(kw)
    return ctk.CTkFrame(padre, **opciones)


class Avatar(ctk.CTkLabel):
    """Círculo con la inicial de la persona."""

    def __init__(self, padre, nombre, colores, tam=36):
        inicial = "?" if nombre.startswith("Desconocido") else nombre[:1].upper()
        super().__init__(padre, text=inicial, image=img_circulo(tam, colores[1]), compound="center", width=tam,
                         height=tam, fg_color="transparent", text_color=C["texto"] if inicial != "?" else C["suave"],
                         font=ctk.CTkFont(familia(), int(tam * 0.38), "bold"))


class Punto(ctk.CTkLabel):
    def __init__(self, padre, color, tam=8):
        super().__init__(padre, text="", image=img_circulo(tam, color), width=tam, height=tam,
                         fg_color="transparent")
        self.tam = tam

    def color(self, color):
        self.configure(image=img_circulo(self.tam, color))


class Onda(tk.Canvas):
    """Dibujo del audio con una rayita que avanza mientras suena."""

    def __init__(self, padre, color, alto=52):
        self.escala = ctk.ScalingTracker.get_widget_scaling(padre)
        super().__init__(padre, height=int(alto * self.escala), bg=C["tarjeta"], highlightthickness=0, bd=0)
        self.color, self.picos, self.dibujado = color, None, None
        self.bind("<Configure>", lambda e: self._dibujar(), add="+")

    def poner(self, audio):
        """audio en None deja solo la línea del centro."""
        if audio is None or len(audio) == 0:
            self.picos = None
        else:
            # con 400 columnas alcanza para cualquier ancho de pantalla
            bloques = np.array_split(np.abs(audio), 400)
            picos = np.array([b.max() if len(b) else 0.0 for b in bloques])
            self.picos = picos / (picos.max() + 1e-9)
        self.dibujado = None
        self._dibujar()

    def _dibujar(self):
        ancho, alto = self.winfo_width(), self.winfo_height()
        if ancho < 10 or (ancho, alto, id(self.picos)) == self.dibujado:
            return
        self.dibujado = (ancho, alto, id(self.picos))
        self.delete("all")
        medio = alto / 2
        if self.picos is None:
            self.create_line(0, medio, ancho, medio, fill=C["barra"], width=max(1, int(2 * self.escala)))
            return
        paso = max(3, int(4 * self.escala))
        grosor = max(1, paso - 1)
        n = max(ancho // paso, 1)
        for i in range(n):
            pico = self.picos[int(i * len(self.picos) / n)]
            h = max(1.0, pico * (alto / 2 - 2))
            self.create_line(i * paso + 1, medio - h, i * paso + 1, medio + h, fill=self.color, width=grosor)
        self.cabezal = self.create_line(0, 0, 0, alto, fill=C["texto"], width=max(1, int(2 * self.escala)),
                                        state="hidden")

    def avance(self, fraccion):
        """fraccion de 0 a 1, o None para esconder la rayita."""
        if self.picos is None or self.dibujado is None:
            return
        if fraccion is None:
            self.itemconfigure(self.cabezal, state="hidden")
            return
        x = fraccion * self.winfo_width()
        self.coords(self.cabezal, x, 0, x, self.winfo_height())
        self.itemconfigure(self.cabezal, state="normal")


# ====================================================================== aplicación

class App(ctk.CTk):
    def __init__(self, motor):
        ctk.set_appearance_mode("light")
        super().__init__(fg_color=C["fondo"])
        self.motor = motor
        self.listo = False
        self.q = queue.Queue()
        self.colores = {}
        self.title("Transcriptor")
        icono = os.path.join(INTERNO, "icono.ico")
        if os.name == "nt" and os.path.exists(icono):
            try:
                self.iconbitmap(default=icono)
            except tk.TclError:
                pass
        # tamaño de la ventana según la pantalla
        escala = ctk.ScalingTracker.get_window_scaling(self)
        ancho_pant, alto_pant = self.winfo_screenwidth() / escala, self.winfo_screenheight() / escala
        ancho, alto = int(min(1200, ancho_pant * 0.94)), int(min(780, alto_pant * 0.86))
        self.geometry(f"{ancho}x{alto}+{int((ancho_pant - ancho) * escala / 2)}+{int(max(alto_pant - alto - 40, 0) * escala / 2)}")
        self.minsize(min(900, ancho), min(600, alto))
        if os.name == "nt" and (ancho_pant < 1250 or alto_pant < 820):
            self.after(50, lambda: self.state("zoomed"))     # pantalla chica: ventana maximizada
        fam = familia()
        self.F = {
            "marca": ctk.CTkFont(fam, 16, "bold"), "titulo": ctk.CTkFont(fam, 24, "bold"),
            "sub": ctk.CTkFont(fam, 14), "normal": ctk.CTkFont(fam, 14), "negrita": ctk.CTkFont(fam, 14, "bold"),
            "chico": ctk.CTkFont(fam, 13), "mini": ctk.CTkFont(fam, 12), "nav": ctk.CTkFont(fam, 14),
            "nav_activo": ctk.CTkFont(fam, 14, "bold"), "seccion": ctk.CTkFont(fam, 16, "bold"),
            "mensaje": ctk.CTkFont(fam, 15), "reloj": ctk.CTkFont(fam, 20, "bold"),
            "lectura": ctk.CTkFont(fam, 26, "bold"), "lectura_larga": ctk.CTkFont(fam, 20, "bold"), "grande": ctk.CTkFont(fam, 18, "bold"),
            "boton": ctk.CTkFont(fam, 14, "bold"), "boton_normal": ctk.CTkFont(fam, 14),
        }
        self.I = {n: _icono(n, C["suave"]) for n in ("chat", "personas", "ajustes", "puntos", "archivo", "atras",
                                                     "copiar", "guardar", "refrescar", "onda")}
        self.I["basura"] = _icono("basura", C["rojo"])
        self.I["basura_suave"] = _icono("basura", C["tenue"])
        self.I["guardar_blanco"] = _icono("guardar", "#FFFFFF")
        self.I.update({n + "_activo": _icono(n, C["texto"]) for n in ("chat", "personas", "ajustes", "onda")})
        self.I["mas"] = _icono("mas", "#FFFFFF", 16)
        self.I["play"] = _icono("play", C["texto2"], 14)
        self.I["parar_chico"] = _icono("parar", C["texto2"], 14)
        self.I["check"] = _icono("check", C["verde"], 16)
        self.I["aviso"] = _icono("aviso", C["ambar"], 16)
        self.I["error"] = _icono("aviso", C["rojo"], 16)
        self.I["marca"] = _icono("onda", "#FFFFFF", 18)
        self.I["atras_suave"] = _icono("atras", C["suave"], 18)
        self.I["grabar"] = _icono("grabar", "#FFFFFF", 18)
        self.I["parar"] = _icono("parar", "#FFFFFF", 16)
        self.I["grabar_g"] = _icono("grabar", "#FFFFFF", 22)
        self.I["parar_g"] = _icono("parar", "#FFFFFF", 20)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._armar_lateral()
        self.contenido = ctk.CTkFrame(self, fg_color=C["fondo"], corner_radius=0)
        self.contenido.grid(row=0, column=1, sticky="nsew")
        self.contenido.grid_rowconfigure(0, weight=1)
        self.contenido.grid_columnconfigure(0, weight=1)
        self.paginas = {
            "conversacion": PaginaConversacion(self.contenido, self),
            "separar": PaginaSeparar(self.contenido, self),
            "personas": PaginaPersonas(self.contenido, self),
            "ajustes": PaginaAjustes(self.contenido, self),
            "enrolar": PaginaEnrolar(self.contenido, self),
        }
        for p in self.paginas.values():
            p.grid(row=0, column=0, sticky="nsew")
        self.actual = None
        self.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.after(40, self._bombear)
        self.mostrar("conversacion")
        self.en_hilo(self._cargar_todo, self._modelos_listos, self._modelos_error)

    def _cargar_todo(self):
        texto = self.motor.cargar(lambda t: self.post(self.estado_modelos, t, "cargando"))
        if self.motor.ajustes.get("traducir"):
            try:
                self.post(self.estado_modelos, "Cargando el traductor...", "cargando")
                self.motor.preparar_traductor()
            except Exception as e:
                self.motor.cambiar_ajuste("traducir", False)
                self.post(self.error, str(e))
        return texto

    # ---------------------------------------------------------------- hilos
    def post(self, funcion, *args, **kwargs):
        """Ejecuta funcion(*args, **kwargs) en el hilo de la interfaz."""
        self.q.put((lambda: funcion(*args, **kwargs), ()))

    def _bombear(self):
        # de a poquito, para que la ventana no se trabe
        limite = time.perf_counter() + 0.03
        try:
            while time.perf_counter() < limite:
                f, args = self.q.get_nowait()
                try:
                    f(*args)
                except Exception as e:      # un error de la interfaz no debe tumbar la app
                    print("Error en la interfaz:", e)
        except queue.Empty:
            pass
        self.after(40, self._bombear)

    def en_hilo(self, trabajo, al_terminar=None, al_error=None):
        def correr():
            try:
                r = trabajo()
            except BaseException as e:
                msg = str(e) or e.__class__.__name__
                self.post(al_error or self.error, msg)
                return
            if al_terminar:
                self.post(al_terminar, r)
        threading.Thread(target=correr, daemon=True).start()

    def error(self, msg):
        messagebox.showerror("Transcriptor", msg, parent=self)

    # ---------------------------------------------------------------- barra lateral
    def _armar_lateral(self):
        self.lateral = ctk.CTkFrame(self, width=232, fg_color=C["tarjeta"], corner_radius=0, border_width=0)
        self.lateral.grid(row=0, column=0, sticky="nsw")
        self.lateral.grid_propagate(False)
        ctk.CTkFrame(self, width=1, fg_color=C["borde"], corner_radius=0).grid(row=0, column=0, sticky="nse")
        marca = ctk.CTkFrame(self.lateral, fg_color="transparent", width=1, height=1)
        marca.pack(fill="x", padx=28, pady=(14, 10))
        ctk.CTkLabel(marca, text="" if self.I["marca"] else "≋", image=self.I["marca"], width=34, height=34,
                     corner_radius=10, fg_color=C["acento"], text_color="#FFFFFF").pack(side="left")
        ctk.CTkLabel(marca, text="Transcriptor", font=self.F["marca"], text_color=C["texto"]).pack(side="left",
                                                                                                padx=10)
        self.nav = {}
        for clave, texto in (("conversacion", "Conversación"), ("separar", "Separar voces"), ("personas", "Personas"),
                             ("ajustes", "Ajustes")):
            icono = {"conversacion": "chat", "separar": "onda"}.get(clave, clave)
            b = ctk.CTkButton(self.lateral, text=texto, anchor="w", height=38, corner_radius=10,
                              image=self.I.get(icono), compound="left", font=self.F["nav"],
                              fg_color="transparent", hover_color=C["gris"], text_color=C["suave"],
                              command=lambda k=clave: self.mostrar(k))
            b.icono = icono
            b.pack(fill="x", padx=18, pady=2)
            self.nav[clave] = b
        pie = ctk.CTkFrame(self.lateral, fg_color=C["fondo"], corner_radius=12)
        pie.pack(side="bottom", fill="x", padx=18, pady=12)
        if self.motor.simulado:
            ctk.CTkLabel(pie, text="MODO SIMULADO", font=self.F["mini"], text_color="#5A4300", fg_color="#F2C14E",
                         corner_radius=6, height=22).pack(anchor="w", padx=12, pady=(12, 0))
        self.lbl_entorno = ctk.CTkLabel(pie, text=self.motor.perfil()["nombre"], font=self.F["negrita"],
                                        text_color=C["texto"])
        # en pantallas muy chicas no cabe; el entorno se ve en Ajustes
        escala = ctk.ScalingTracker.get_window_scaling(self)
        if self.winfo_screenheight() / escala >= 560:
            ctk.CTkLabel(pie, text="Entorno", font=self.F["mini"], text_color=C["tenue"]).pack(anchor="w", padx=14,
                                                                                             pady=(12, 0))
            self.lbl_entorno.pack(anchor="w", padx=14)
        ctk.CTkLabel(pie, text="Equipo", font=self.F["mini"], text_color=C["tenue"]).pack(anchor="w", padx=14,
                                                                                        pady=(8, 0))
        fila = ctk.CTkFrame(pie, fg_color="transparent", width=1, height=1)
        fila.pack(fill="x", padx=14, pady=(0, 12))
        self.punto = Punto(fila, "#E0A800", 8)
        self.punto.pack(side="left", padx=(0, 7))
        self.lbl_equipo = ctk.CTkLabel(fila, text="Cargando...", font=self.F["negrita"], text_color=C["texto"],
                                       anchor="w", justify="left", wraplength=160)
        self.lbl_equipo.pack(side="left", fill="x")

    def mostrar(self, clave):
        if self.actual == "enrolar" and clave != "enrolar" and self.paginas["enrolar"].grabando:
            return
        if clave == "enrolar" and self.motor.en_curso():
            messagebox.showinfo("Transcriptor", "Detén la conversación antes de enrolar a alguien.", parent=self)
            return
        if self.actual == "separar" and clave != "separar":
            self.paginas["separar"].parar()      # que no se siga oyendo en otra pantalla
        self.actual = clave
        # el enrolamiento es una pantalla enfocada: sin barra lateral
        if clave == "enrolar":
            self.lateral.grid_remove()
        else:
            self.lateral.grid()
        for k, b in self.nav.items():
            activo = k == clave
            b.configure(fg_color=C["acento_suave"] if activo else "transparent",
                        text_color=C["texto"] if activo else C["suave"],
                        font=self.F["nav_activo"] if activo else self.F["nav"],
                        image=self.I.get(b.icono + ("_activo" if activo else "")))
        self.paginas[clave].tkraise()
        self.paginas[clave].al_mostrar()

    def estado_modelos(self, texto, estado):
        self.lbl_equipo.configure(text=texto, font=self.F["chico"] if estado == "cargando" else self.F["negrita"])
        self.punto.color({"listo": "#2E9E62", "error": C["rojo"]}.get(estado, "#E0A800"))

    def _modelos_listos(self, descripcion):
        print("Modelos listos:", descripcion)
        self.listo = True
        self.estado_modelos(self.motor.equipo(), "listo")
        for p in self.paginas.values():
            p.al_cambiar_modelos()

    def _modelos_error(self, msg):
        self.estado_modelos("No se pudieron cargar los modelos", "error")
        self.error("No se pudieron cargar los modelos:\n\n" + msg +
                   "\n\nRevisa que el entorno virtual esté activado (.venv) y que la instalación esté completa.")

    def entorno_cambio(self):
        self.lbl_entorno.configure(text=self.motor.perfil()["nombre"])

    def personas_cambiaron(self):
        for p in self.paginas.values():
            p.al_cambiar_personas()

    def color(self, nombre):
        """Cada persona conserva su color en toda la app (el orden lo da la lista de enrolados)."""
        if nombre.startswith("Desconocido"):
            return COLOR_DESCONOCIDO
        if nombre not in self.colores:
            self.colores[nombre] = COLORES_PERSONA[len(self.colores) % len(COLORES_PERSONA)]
        return self.colores[nombre]

    def cerrar(self):
        conv = self.paginas["conversacion"]
        if self.motor.en_curso() or conv.ocupado or self.paginas["separar"].ocupado:
            if not messagebox.askyesno("Transcriptor", "Hay una grabación o transcripción en curso. "
                                                       "¿Salir de todos modos?", parent=self):
                return
            try:
                self.motor.detener()
            except Exception:
                pass
        self.destroy()

    # ---------------------------------------------------------------- piezas comunes
    def encabezado(self, padre, titulo, subtitulo):
        f = ctk.CTkFrame(padre, fg_color="transparent", width=1, height=1)
        izq = ctk.CTkFrame(f, fg_color="transparent", width=1, height=1)
        izq.pack(side="left", anchor="w")
        ctk.CTkLabel(izq, text=titulo, font=self.F["titulo"], text_color=C["texto"], anchor="w").pack(anchor="w")
        f.subtitulo = ctk.CTkLabel(izq, text=subtitulo, font=self.F["sub"], text_color=C["tenue"], anchor="w")
        f.subtitulo.pack(anchor="w")
        f.derecha = ctk.CTkFrame(f, fg_color="transparent", width=1, height=1)
        f.derecha.pack(side="right")
        return f

    def microfonos(self):
        """[(id, nombre)] y el que está elegido en Ajustes (o el predeterminado)."""
        try:
            lista, pred = self.motor.microfonos()
        except Exception:
            return [], None
        guardado = self.motor.ajustes.get("microfono")
        for i, n in lista:
            if n == guardado:
                return lista, i
        return lista, pred

    def microfono_elegido(self):
        lista, elegido = self.microfonos()
        return elegido


class Pagina(ctk.CTkFrame):
    def __init__(self, padre, app):
        super().__init__(padre, fg_color=C["fondo"], corner_radius=0)
        self.app = app

    def al_mostrar(self):
        pass

    def al_cambiar_modelos(self):
        pass

    def al_cambiar_personas(self):
        pass


# ====================================================================== separar voces

class PaginaSeparar(Pagina):
    """Abre un audio corto con dos personas hablando a la vez y deja oír cada voz por separado."""
    MAX_SEG = 9.5       # MossFormer2 no aguanta pedazos más largos

    def __init__(self, padre, app):
        super().__init__(padre, app)
        self.ocupado = False
        self.audio = self.nombre_archivo = None
        self.sonando = None
        self._reloj = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        enc = app.encabezado(self, "Separar voces", "Cada voz en su propia pista.")
        enc.grid(row=0, column=0, sticky="ew", padx=28, pady=(20, 12))
        self.btn_abrir = ctk.CTkButton(enc.derecha, text="Abrir audio", image=app.I["archivo"], compound="left", height=40,
                                       corner_radius=10, font=app.F["boton_normal"], fg_color=C["gris"],
                                       hover_color=C["gris_hover"], text_color=C["texto"], command=self.abrir)
        self.btn_abrir.pack(side="left", padx=(0, 10))
        self.btn_separar = ctk.CTkButton(enc.derecha, text="Separar", height=40, corner_radius=10,
                                         font=app.F["boton"], fg_color=C["acento"], hover_color=C["acento_hover"],
                                         text_color="#FFFFFF", text_color_disabled="#FFFFFF", command=self.separar,
                                         state="disabled")
        self.btn_separar.pack(side="left")

        # tarjeta de la mezcla
        t = self._tarjeta(self)
        t.grid(row=1, column=0, sticky="ew", padx=28)
        cab = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        cab.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 6))
        cab.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(cab, text="Audio original", font=app.F["negrita"], text_color=C["texto"], height=22,
                     anchor="w").grid(row=0, column=0, sticky="w")
        self.lbl_archivo = ctk.CTkLabel(cab, text="Abre un audio de hasta 9.5 segundos para empezar.", height=18,
                                        font=app.F["chico"], text_color=C["tenue"], anchor="w", justify="left")
        self.lbl_archivo.grid(row=1, column=0, sticky="w")
        ajustar_texto(self.lbl_archivo, cab, 150)
        self.mezcla = self._controles(cab, t, "mezcla", "#8A8983")
        self.lbl_mezcla = ctk.CTkLabel(t, text="", font=app.F["chico"], text_color=C["suave"], anchor="w", justify="left")
        ajustar_texto(self.lbl_mezcla, t, 44)

        # avance mientras trabaja
        self.progreso = ctk.CTkFrame(self, fg_color="transparent", width=1, height=1)
        self.progreso.grid_columnconfigure(0, weight=1)
        self.lbl_progreso = ctk.CTkLabel(self.progreso, text="", font=app.F["chico"], text_color=C["acento_hover"], height=20,
                                         anchor="w")
        self.lbl_progreso.grid(row=0, column=0, sticky="w")
        self.barra = ctk.CTkProgressBar(self.progreso, height=6, corner_radius=3, fg_color=C["barra"],
                                        progress_color=C["acento"])
        self.barra.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        # las dos pistas, lado a lado
        pistas = ctk.CTkFrame(self, fg_color="transparent", width=1, height=1)
        pistas.grid(row=3, column=0, sticky="nsew", padx=28, pady=(14, 0))
        pistas.grid_rowconfigure(0, weight=1)
        pistas.grid_columnconfigure((0, 1), weight=1, uniform="pistas")
        self.pistas = [self._pista(pistas, i) for i in range(2)]

        self.lbl_tiempo = ctk.CTkLabel(self, text="", font=app.F["chico"], text_color=C["tenue"], height=20, anchor="w",
                                       justify="left")
        self.lbl_tiempo.grid(row=4, column=0, sticky="ew", padx=30, pady=(8, 16))
        ajustar_texto(self.lbl_tiempo, self, 60)
        self._estrecho = False
        self._limpiar_pistas()
        self.bind("<Configure>", self._acomodar, add="+")

    def _acomodar(self, _=None):
        """En pantallas chicas (o con escala de Windows) las ondas se hacen bajitas y los botones solo llevan ícono."""
        escala = ctk.ScalingTracker.get_widget_scaling(self)
        ancho, alto = self.winfo_width() / escala, self.winfo_height() / escala
        estrecho = ancho < 800
        altura = int((30 if alto < 600 else 52) * escala)
        for c in [self.mezcla] + self.pistas:
            if int(c["onda"].cget("height")) != altura:
                c["onda"].configure(height=altura)
        if estrecho != self._estrecho:
            self._estrecho = estrecho
            for i, p in enumerate(self.pistas):
                self._rotular(p, self.sonando == i)

    # ---------------------------------------------------------------- piezas
    def _tarjeta(self, padre):
        t = ctk.CTkFrame(padre, fg_color=C["tarjeta"], corner_radius=14, border_width=1, border_color=C["borde"])
        t.grid_columnconfigure(0, weight=1)
        return t

    def _controles(self, cab, tarjeta, clave, color):
        """Botón de escuchar (en el encabezado de la tarjeta) y dibujo del audio (abajo de él)."""
        boton = ctk.CTkButton(cab, text="Escuchar", image=self.app.I["play"], compound="left", width=112, height=34,
                              corner_radius=9, font=self.app.F["boton_normal"], fg_color=C["gris"],
                              hover_color=C["gris_hover"], text_color=C["texto"], state="disabled",
                              text_color_disabled=C["apagado"], command=lambda: self.escuchar(clave))
        boton.grid(row=0, column=2, rowspan=2, sticky="e", padx=(10, 0))
        onda = Onda(tarjeta, color)
        onda.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 12))
        return {"boton": boton, "onda": onda, "audio": None}

    def _pista(self, padre, i):
        t = self._tarjeta(padre)
        t.grid(row=0, column=i, sticky="nsew", padx=(0, 7) if i == 0 else (7, 0))
        cab = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        cab.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 6))
        cab.grid_columnconfigure(1, weight=1)
        hueco = ctk.CTkFrame(cab, fg_color="transparent", width=40, height=40)
        hueco.grid(row=0, column=0, rowspan=2, padx=(0, 10))
        nombre = ctk.CTkLabel(cab, text="", font=self.app.F["negrita"], text_color=C["texto"], height=22, anchor="w")
        nombre.grid(row=0, column=1, sticky="w")
        detalle = ctk.CTkLabel(cab, text="", font=self.app.F["mini"], text_color=C["tenue"], height=18, anchor="w")
        detalle.grid(row=1, column=1, sticky="w")
        c = self._controles(cab, t, i, COLOR_DESCONOCIDO[0])
        texto = ctk.CTkLabel(t, text="", font=self.app.F["mensaje"], text_color=C["texto2"], anchor="nw", justify="left")
        texto.grid(row=2, column=0, sticky="new", padx=18, pady=(0, 16))
        ajustar_texto(texto, t, 44)
        c.update(tarjeta=t, hueco=hueco, nombre=nombre, detalle=detalle, texto=texto, avatar=None)
        return c

    def _poner_avatar(self, p, nombre, colores):
        if p["avatar"] is not None:
            p["avatar"].destroy()
        p["avatar"] = Avatar(p["hueco"], nombre, colores, 36)
        p["avatar"].place(relx=0.5, rely=0.5, anchor="center")

    def _rotular(self, c, sonando):
        """Texto e ícono del botón. Las pistas en pantalla angosta solo llevan el ícono."""
        solo_icono = self._estrecho and c is not self.mezcla
        c["boton"].configure(text="" if solo_icono else ("Detener" if sonando else "Escuchar"), width=40 if solo_icono else 112,
                             image=self.app.I["parar_chico" if sonando else "play"])

    def _limpiar_pistas(self):
        for i, p in enumerate(self.pistas):
            p["audio"] = None
            p["onda"].color = COLOR_DESCONOCIDO[0]
            p["onda"].poner(None)
            p["boton"].configure(state="disabled")
            self._poner_avatar(p, str(i + 1), COLOR_DESCONOCIDO)
            p["nombre"].configure(text=f"Pista {i + 1}")
            p["detalle"].configure(text="Todavía sin separar")
            p["texto"].configure(text="", text_color=C["texto2"])
        self.lbl_mezcla.grid_forget()
        self.lbl_tiempo.configure(text="")

    def _botones(self):
        libre = not self.ocupado
        self.btn_abrir.configure(state="normal" if libre else "disabled")
        puede = libre and self.audio is not None and self.app.listo
        self.btn_separar.configure(state="normal" if puede else "disabled",
                                   fg_color=C["acento"] if puede else C["acento_borde"])

    def al_cambiar_modelos(self):
        self._botones()

    # ---------------------------------------------------------------- abrir
    def abrir(self):
        ruta = filedialog.askopenfilename(
            parent=self.app, title="Elige un audio con dos voces encimadas",
            initialdir=os.path.join(DATOS, "transcripciones") if os.path.isdir("transcripciones") else DATOS,
            filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.webm *.aac"), ("Todos", "*.*")])
        if ruta:
            self.cargar(ruta)

    def cargar(self, ruta):
        self.parar()
        self.ocupado = True
        self._botones()
        self.app.en_hilo(lambda: mt.tr.cargar_audio(ruta), lambda x: self._cargado(x, ruta), self._fallo_abrir)

    def _fallo_abrir(self, msg):
        self.ocupado = False
        self._botones()
        self.app.error(f"No se pudo abrir el audio:\n\n{msg}")

    def _cargado(self, x, ruta):
        self.ocupado = False
        duracion = len(x) / mt.SR
        if duracion < 1:
            self._botones()
            self.app.error("El audio es demasiado corto. Elige uno de al menos 1 segundo.")
            return
        recortado = duracion > self.MAX_SEG
        self.audio = x[:int(self.MAX_SEG * mt.SR)]
        texto = f"{os.path.basename(ruta)}  ·  {len(self.audio) / mt.SR:.1f} s"
        if recortado:     # el separador no aguanta más, mejor avisar que cortar sin decir nada
            texto += f"  ·  el audio dura {duracion:.0f} s y solo se usan los primeros {self.MAX_SEG} s"
        self.lbl_archivo.configure(text=texto)
        self.mezcla["audio"] = self.audio
        self.mezcla["onda"].poner(self.audio)
        self.mezcla["boton"].configure(state="normal")
        self._limpiar_pistas()
        self._botones()

    # ---------------------------------------------------------------- separar
    def separar(self):
        conv = self.app.paginas["conversacion"]
        if self.app.motor.en_curso() or conv.ocupado:
            messagebox.showinfo("Transcriptor", "Espera a que termine la grabación o la transcripción.", parent=self.app)
            return
        self.parar()
        self._limpiar_pistas()
        self.ocupado = True
        self._botones()
        self.barra.set(0)
        self.lbl_progreso.configure(text="Preparando...")
        self.progreso.grid(row=2, column=0, sticky="ew", padx=30, pady=(12, 0))
        audio = self.audio
        self.app.en_hilo(lambda: self.app.motor.separar_audio(audio, lambda t, f: self.app.post(self._avance, t, f)),
                         self._terminado, self._fallo)

    def _avance(self, texto, fraccion):
        self.lbl_progreso.configure(text=texto + "...")
        self.barra.set(fraccion)

    def _fallo(self, msg):
        self.ocupado = False
        self.progreso.grid_forget()
        self._botones()
        self.app.error(f"No se pudieron separar las voces:\n\n{msg}")

    def _terminado(self, r):
        self.ocupado = False
        self.progreso.grid_forget()
        self._botones()
        for i, (p, d) in enumerate(zip(self.pistas, r["pistas"])):
            colores = self.app.color(d["nombre"])
            p["audio"] = d["audio"]
            p["onda"].color = colores[0]
            p["onda"].poner(d["audio"])
            p["boton"].configure(state="normal")
            self._poner_avatar(p, d["nombre"], colores)
            p["nombre"].configure(text=d["nombre"])
            extra = f"  ·  parecido {d['parecido']:.2f}" if d["parecido"] is not None else ""
            p["detalle"].configure(text=f"Pista {i + 1}{extra}")
            if d["texto"]:
                p["texto"].configure(text=d["texto"], text_color=C["texto2"])
            else:
                p["texto"].configure(text="No se entendió ninguna frase en esta pista.", text_color=C["tenue"])
        if r["texto_mezcla"]:
            self.lbl_mezcla.configure(text=f"Sin separar, Whisper escribió: «{r['texto_mezcla']}»")
        else:
            self.lbl_mezcla.configure(text="Sin separar, Whisper no escribió nada.")
        self.lbl_mezcla.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        duracion = len(self.audio) / mt.SR
        trabajo = r["separar"] + r["transcribir"]
        self.lbl_tiempo.configure(text=(
            f"Tardó {_seg(trabajo)} con {r['equipo']} ({trabajo / duracion:.1f} veces lo que dura el audio): "
            f"separar {_seg(r['separar'])}, transcribir {_seg(r['transcribir'])}"
            + (f". Cargar modelos: {_seg(r['cargar'])}." if r["cargar"] >= 1 else ".")))

    # ---------------------------------------------------------------- escuchar
    def _control(self, clave):
        return self.mezcla if clave == "mezcla" else self.pistas[clave]

    def escuchar(self, clave):
        if self.sonando == clave:
            self.parar()
            return
        self.parar()
        c = self._control(clave)
        if c["audio"] is None:
            return
        try:
            self.app.motor.reproducir(c["audio"])
        except Exception as e:
            self.app.error(f"No se pudo reproducir el audio:\n\n{e}")
            return
        self.sonando = clave
        self._rotular(c, True)
        self._inicio = time.time()
        self._mover()

    def _mover(self):
        c = self._control(self.sonando)
        fraccion = (time.time() - self._inicio) / (len(c["audio"]) / mt.SR)
        if fraccion >= 1:
            self.parar()
            return
        c["onda"].avance(fraccion)
        self._reloj = self.after(40, self._mover)

    def parar(self):
        if self.sonando is None:
            return
        c = self._control(self.sonando)
        self.sonando = None
        if self._reloj is not None:
            self.after_cancel(self._reloj)
            self._reloj = None
        try:
            self.app.motor.parar_audio()
        except Exception as e:
            print("No se pudo parar el audio:", e)
        self._rotular(c, False)
        c["onda"].avance(None)


def _seg(s):
    return f"{s:.0f} s" if s >= 10 else f"{s:.1f} s"


# ====================================================================== conversación

def _duracion_audio(ruta):
    try:
        import soundfile as sf
        return sf.info(ruta).duration
    except Exception:
        return 60.0


def _texto_aprox(segundos):
    if segundos is None:
        return ""
    return f"  ·  aprox. {max(1, round(segundos / 60))} min en total" if segundos >= 60 else f"  ·  aprox. {max(1, round(segundos))} s en total"


class PaginaConversacion(Pagina):
    def __init__(self, padre, app):
        super().__init__(padre, app)
        self.lineas = {"vivo": [], "final": []}
        self.vista = "vivo"
        self.filas = []
        self.ocupado = False
        self.inicio = None
        self.ruta_audio = None
        self.forzar_final = False
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.cab = app.encabezado(self, "Conversación", "Cargando modelos...")
        self.cab.grid(row=0, column=0, sticky="ew", padx=40, pady=(28, 18))
        self.chips = ctk.CTkFrame(self.cab.derecha, fg_color="transparent", width=1, height=1)
        self.chips.pack(side="right")
        self.selector = ctk.CTkSegmentedButton(self.cab.derecha, values=["En vivo", "Final"],
                                               command=self._cambiar_vista, font=self.F_("chico"), height=32,
                                               selected_color=C["tarjeta"], selected_hover_color=C["tarjeta"],
                                               unselected_color=C["gris_hover"], unselected_hover_color=C["barra"],
                                               fg_color=C["gris_hover"], text_color=C["texto"], corner_radius=8)

        self.hoja = tarjeta(self, corner_radius=18)
        self.hoja.grid(row=1, column=0, sticky="nsew", padx=40)
        self.hoja.grid_columnconfigure(0, weight=1)
        self.hoja.grid_rowconfigure(1, weight=1)
        self.progreso = ctk.CTkFrame(self.hoja, fg_color="transparent", width=1, height=1)
        self.btn_cancelar = boton(self.progreso, "Cancelar", self.cancelar, "secundario", font=self.F_("chico"), height=32,
                                  width=90)
        self.btn_cancelar.pack(side="right", padx=(14, 0))
        self.lbl_progreso = ctk.CTkLabel(self.progreso, text="", font=self.F_("chico"), text_color=C["acento_hover"],
                                         anchor="w")
        self.lbl_progreso.pack(fill="x")
        self.barra_prog = ctk.CTkProgressBar(self.progreso, height=6, progress_color=C["acento"],
                                             fg_color=C["barra"], corner_radius=3)
        self.barra_prog.pack(fill="x", pady=(4, 0))
        self.barra_prog.set(0)
        self.lista = ctk.CTkScrollableFrame(self.hoja, fg_color=C["tarjeta"], corner_radius=0,
                                            scrollbar_button_color=C["barra"],
                                            scrollbar_button_hover_color=C["apagado"])
        self.lista.grid(row=1, column=0, sticky="nsew", padx=(14, 6), pady=14)
        self.lista.grid_columnconfigure(0, weight=1)
        self.hoja.bind("<Configure>", self._al_redimensionar, add="+")
        self.vacio = ctk.CTkLabel(self.lista, text="", font=self.F_("sub"), text_color=C["tenue"], justify="center")
        self.escuchando = ctk.CTkFrame(self.lista, fg_color="transparent", width=1, height=1)
        self.puntos_hablando = [Punto(self.escuchando, C["barra"], 6) for _ in range(3)]
        for p in self.puntos_hablando:
            p.pack(side="left", padx=2)
        ctk.CTkLabel(self.escuchando, text="  Escuchando", font=self.F_("chico"), text_color=C["tenue"]).pack(
            side="left")

        pie = tarjeta(self, corner_radius=18)
        self.pie = pie
        self.compacto = None
        pie.bind("<Configure>", self._acomodar_pie, add="+")
        pie.grid(row=2, column=0, sticky="ew", padx=40, pady=(18, 30))
        izq = ctk.CTkFrame(pie, fg_color="transparent", width=1, height=1)
        izq.pack(side="left", padx=16, pady=12)
        self.btn_grabar = BotonRedondo(izq, 52, self.alternar)
        self.btn_grabar.pack(side="left")
        textos = ctk.CTkFrame(izq, fg_color="transparent", width=1, height=1)
        textos.pack(side="left", padx=14)
        self.lbl_reloj = ctk.CTkLabel(textos, text="00:00", font=self.F_("reloj"), text_color=C["texto"],
                                      anchor="w", height=26)
        self.lbl_reloj.pack(anchor="w")
        self.lbl_estado = ctk.CTkLabel(textos, text="Esperando modelos...", font=self.F_("chico"),
                                       text_color=C["tenue"], anchor="w", height=18)
        self.lbl_estado.pack(anchor="w")
        self.medidor = Medidor(pie, barras=14)
        self.medidor.pack(side="left", padx=10)
        der = ctk.CTkFrame(pie, fg_color="transparent", width=1, height=1)
        self.der_pie = der
        der.pack(side="right", padx=16)
        self.btn_terminar = boton(der, "Terminar y mejorar texto", self.terminar_y_mejorar, "principal",
                                  font=self.F_("boton"), height=44)
        self.btn_descartar = boton(der, "Descartar", self.descartar, "peligro", font=self.F_("boton_normal"), height=44,
                                   image=app.I["basura"], width=140)
        self.btn_abrir = boton(der, "Abrir grabación", self.abrir_audio, "secundario", font=self.F_("boton_normal"),
                               height=44, image=app.I["archivo"])
        self.btn_copiar = boton(der, "Copiar", self.copiar, "secundario", font=self.F_("boton_normal"), height=44,
                                image=app.I["copiar"])
        self.btn_guardar = boton(der, "Guardar", self.guardar, "secundario", font=self.F_("boton_normal"),
                                 height=44, image=app.I["guardar"])
        self.btn_limpiar = boton(der, "", self.limpiar, "secundario", width=44, height=44, image=app.I["basura"],
                                 hover_color=C["rojo_suave"])
        self.btn_abrir.pack(side="left", padx=(0, 10))
        self.btn_copiar.pack(side="left", padx=(0, 10))
        self.btn_guardar.pack(side="left", padx=(0, 10))
        self.btn_limpiar.pack(side="left")
        self.redibujar()
        self._actualizar_botones()
        self._tic()

    def F_(self, k):
        return self.app.F[k]

    def _acomodar_pie(self, e=None):
        """En ventanas angostas los botones de abajo se quedan solo con su ícono."""
        escala = ctk.ScalingTracker.get_widget_scaling(self.pie)
        compacto = self.pie.winfo_width() / escala < 860
        if compacto == self.compacto:
            return
        self.compacto = compacto
        for b, texto in ((self.btn_abrir, "Abrir grabación"), (self.btn_copiar, "Copiar"), (self.btn_guardar, "Guardar"),
                         (self.btn_descartar, "Descartar")):
            b.configure(text="" if compacto else texto, width=44 if compacto else 140)
        self.btn_terminar.configure(text="Terminar" if compacto else "Terminar y mejorar texto", width=100 if compacto else 140)
        if compacto:
            self.medidor.pack_forget()
        else:
            self.medidor.pack(side="left", padx=10, before=self.der_pie)

    # ---------------------------------------------------------------- estado
    def al_cambiar_modelos(self):
        self._actualizar_botones()
        self.al_cambiar_personas()
        self.redibujar()

    def al_cambiar_personas(self):
        for w in self.chips.winfo_children():
            w.destroy()
        if not self.app.listo:
            return
        try:
            nombres = [p["nombre"] for p in self.app.motor.personas()]
        except Exception:
            nombres = []
        for n in nombres[:5]:       # tocar un nombre lo quita o lo regresa a la plática
            si = self.app.motor.presente(n)
            chip = ctk.CTkFrame(self.chips, fg_color=C["tarjeta"] if si else C["gris"], corner_radius=16, border_width=1,
                                border_color=C["borde"], height=32, cursor="hand2")
            chip.pack(side="left", padx=(8, 0))
            punto = Punto(chip, self.app.color(n)[0] if si else C["apagado"])
            punto.pack(side="left", padx=(12, 6), pady=11)
            nombre = ctk.CTkLabel(chip, text=n, font=self.F_("chico"), text_color=C["texto"] if si else C["apagado"])
            nombre.pack(side="left", padx=(0, 12))
            for w in (chip, punto, nombre):
                w.configure(cursor="hand2")
                w.bind("<Button-1>", lambda e, x=n, v=not si: self.cambiar_presencia(x, v), add="+")
        if len(nombres) > 5:
            ctk.CTkLabel(self.chips, text=f"+{len(nombres) - 5}", font=self.F_("chico"),
                         text_color=C["tenue"]).pack(side="left", padx=8)
        self._subtitulo()

    def cambiar_presencia(self, nombre, si):
        """Regresa True si se pudo. Mientras se graba o se procesa no se cambia, para no mezclar."""
        motor = self.app.motor
        if motor.en_curso() or self.ocupado:
            messagebox.showinfo("Transcriptor", "Espera a que termine la grabación para cambiar quién está.", parent=self.app)
            return False
        if not si:
            quedan = [p["nombre"] for p in motor.personas() if motor.presente(p["nombre"]) and p["nombre"] != nombre]
            if not quedan:
                messagebox.showinfo("Transcriptor", "Debe quedar al menos una persona en la plática.", parent=self.app)
                return False
        motor.poner_presente(nombre, si)
        self.al_cambiar_personas()
        return True

    def _subtitulo(self):
        if not self.app.listo:
            texto = "Cargando modelos..."
        else:
            personas = self.app.motor.personas()
            n = len(personas)
            aqui = sum(self.app.motor.presente(p["nombre"]) for p in personas)
            quien = f"{n} persona registrada" if n == 1 else f"{n} personas registradas"
            if aqui < n:
                quien = f"{aqui} de {n} en la plática"
            if n == 0:
                quien = "Nadie registrado: todas las voces saldrán como Desconocido"
            texto = ("Escuchando · " if self.app.motor.en_curso() else "") + quien
        self.cab.subtitulo.configure(text=texto)

    def _actualizar_botones(self):
        listo, grabando = self.app.listo, self.app.motor.en_curso()
        self.btn_grabar.poner(activo=listo and not self.ocupado)
        self.btn_abrir.configure(state="normal" if listo and not grabando and not self.ocupado else "disabled")
        hay = bool(self.lineas[self.vista])
        self.btn_copiar.configure(state="normal" if hay else "disabled")
        self.btn_guardar.configure(state="normal" if hay and not grabando else "disabled")
        self.btn_limpiar.configure(state="normal" if hay and not grabando and not self.ocupado else "disabled")
        if grabando:
            for b in (self.btn_abrir, self.btn_guardar, self.btn_limpiar):
                b.pack_forget()
            self.btn_terminar.pack(side="left", padx=(0, 10), before=self.btn_copiar)
            self.btn_descartar.pack(side="left", padx=(0, 10), before=self.btn_copiar)
            self.btn_descartar.configure(state="disabled" if self.ocupado else "normal")
        else:
            self.btn_terminar.pack_forget()
            self.btn_descartar.pack_forget()
            self.btn_abrir.pack(side="left", padx=(0, 10), before=self.btn_copiar)
            self.btn_guardar.pack(side="left", padx=(0, 10), after=self.btn_copiar)
            self.btn_limpiar.pack(side="left", after=self.btn_guardar)
        if listo and not grabando and not self.ocupado:
            self.lbl_estado.configure(text="Listo para grabar", text_color=C["tenue"])
        self._subtitulo()

    def _tic(self):
        if self.inicio is not None:
            self.lbl_reloj.configure(text=mmss(time.time() - self.inicio))
        self.after(250, self._tic)

    # ---------------------------------------------------------------- grabar / detener
    def alternar(self):
        if self.app.motor.en_curso():
            self.detener()
        else:
            self.grabar()

    def _separando(self):
        if self.app.paginas["separar"].ocupado:
            messagebox.showinfo("Transcriptor", "Espera a que termine de separar las voces.", parent=self.app)
            return True
        return False

    def grabar(self):
        if self._separando():
            return
        if (self.lineas["vivo"] or self.lineas["final"]) and not messagebox.askyesno(
                "Nueva conversación", "Se borrará la transcripción que está en pantalla. ¿Continuar?",
                parent=self.app):
            return
        self.lineas = {"vivo": [], "final": []}
        self.vista = "vivo"
        self.redibujar()
        try:
            self.ruta_audio = self.app.motor.iniciar(self.app.microfono_elegido(), self._evento_motor,
                                                     espanglish=self.app.motor.ajustes.get("espanglish", True))
        except Exception as e:
            self.app.error(f"No se pudo empezar a grabar:\n\n{e}")
            return
        self.inicio = time.time()
        self.forzar_final = False
        self.btn_grabar.poner(modo="parar")
        self.lbl_estado.configure(text="Grabando · " + self.app.motor.perfil()["nombre"].lower(),
                                  text_color=C["rojo"])
        self._actualizar_botones()
        self.redibujar()

    def _evento_motor(self, tipo, datos):
        # llega desde los hilos del motor: se pasa al hilo de la interfaz
        if tipo == "nivel":
            self.app.post(self._nivel, datos["rms"], datos["hablando"])
        elif tipo == "linea":
            self.app.post(self.agregar_linea, datos, "vivo")
        elif tipo == "aviso":
            self.app.post(self._aviso, datos)
        elif tipo == "atraso":
            self.app.post(self._atraso, datos)

    def _nivel(self, rms, hablando):
        self.medidor.poner(rms, hablando)
        fase = int(time.time() * 3) % 3
        for i, p in enumerate(self.puntos_hablando):
            p.color(C["apagado"] if hablando and i == fase else C["barra"])

    def _atraso(self, segundos):
        if not self.app.motor.en_curso():
            return
        if segundos > 3:
            self.lbl_estado.configure(text=f"Grabando · el texto va {segundos:.0f} s atrás", text_color=C["ambar"])
        else:
            self.lbl_estado.configure(text="Grabando · " + self.app.motor.perfil()["nombre"].lower(), text_color=C["rojo"])

    def _aviso(self, mensaje):
        self.lbl_estado.configure(text="Revisa el aviso", text_color=C["ambar"])
        messagebox.showwarning("Transcriptor", mensaje, parent=self.app)

    def terminar_y_mejorar(self):
        self.forzar_final = True
        self.detener()

    def detener(self):
        self.ocupado = True
        self.btn_grabar.poner(activo=False)
        self.btn_terminar.configure(state="disabled")
        self.lbl_estado.configure(text="Terminando lo pendiente...", text_color=C["tenue"])
        self.app.en_hilo(self.app.motor.detener, self._detenido, self._fallo_detener)

    def descartar(self):
        if not messagebox.askyesno("Descartar grabación", "Se borrará el audio y el texto de esta grabación. "
                                   "No se puede deshacer. ¿Continuar?", parent=self.app):
            return
        self.ocupado = True
        self.btn_grabar.poner(activo=False)
        self.btn_terminar.configure(state="disabled")
        self.btn_descartar.configure(state="disabled")
        self.lbl_estado.configure(text="Descartando...", text_color=C["tenue"])
        self.app.en_hilo(lambda: self.app.motor.detener(descartar=True), self._descartado, self._fallo_detener)

    def _descartado(self, _):
        self.lineas = {"vivo": [], "final": []}
        self.ruta_audio = None
        self._restaurar_boton()
        self.lbl_reloj.configure(text="00:00")
        self.lbl_estado.configure(text="Grabación descartada", text_color=C["tenue"])

    def _fallo_detener(self, msg):
        self._restaurar_boton()
        self.app.error(msg)

    def _restaurar_boton(self):
        self.inicio = None
        self.ocupado = False
        self.btn_grabar.poner(modo="grabar")
        self.btn_terminar.configure(state="normal")
        self.medidor.reiniciar()
        self._actualizar_botones()
        self.redibujar()

    def _detenido(self, ruta):
        self._restaurar_boton()
        self.ruta_audio = ruta
        if ruta and (self.forzar_final or self.app.motor.ajustes["modo_final"] != "rapido"):
            self.correr_pasada_final(ruta)
        elif ruta:
            self.app.motor.registrar_rapido(ruta)

    # ---------------------------------------------------------------- versión final
    def correr_pasada_final(self, ruta, titulo="Mejorando el texto"):
        self.ocupado = True
        self._actualizar_botones()
        self.lbl_estado.configure(text=f"{titulo}...", text_color=C["acento_hover"])
        self.progreso.grid(row=0, column=0, sticky="ew", padx=28, pady=(18, 0))
        self.barra_prog.set(0)
        self.btn_cancelar.configure(state="normal")
        modo = self.app.motor.ajustes["modo_final"]
        self._aprox = _texto_aprox(self.app.motor.estimar(_duracion_audio(ruta), "mejorado" if modo == "rapido" else modo))
        self.lbl_progreso.configure(text=f"{titulo}: preparando...{self._aprox}")

        def avance(etapa, fraccion):
            if "encimadas" in etapa:
                self.app.post(self._avance, f"{titulo}: separando voces encimadas… {int(fraccion * 100)} %",
                              0.95 + 0.05 * fraccion)
                return
            if etapa.startswith("Cargando"):
                self.app.post(self._avance, f"{titulo}: {etapa.lower()} (solo la primera vez, unos segundos)…", 0.0)
                return
            total = 0.3 * fraccion if etapa.startswith("Separando") else 0.3 + 0.7 * fraccion
            self.app.post(self._avance, f"{titulo}: {etapa.lower()}… {int(total * 100)} %", total)

        espanglish = self.app.motor.ajustes.get("espanglish", True)
        self.app.en_hilo(lambda: self.app.motor.pasada_final(ruta, avance, espanglish),
                         self._final_listo, self._final_error)

    def cancelar(self):
        self.btn_cancelar.configure(state="disabled")
        self.lbl_progreso.configure(text="Cancelando... (termina el pedazo en el que va)")
        self.app.motor.cancelar()

    def _avance(self, texto, valor):
        if self.btn_cancelar.cget("state") == "disabled":
            return      # ya se pidió cancelar: que no se borre el aviso
        self.lbl_progreso.configure(text=texto + self._aprox)
        self.barra_prog.set(valor)

    def _final_listo(self, lineas):
        self.progreso.grid_forget()
        self.ocupado = False
        self.lineas["final"] = []
        for l in lineas:
            self._juntar(self.lineas["final"], l)
        if lineas:
            self.vista = "final"
        self.redibujar(arriba=True)
        self._actualizar_botones()
        m = getattr(self.app.motor, "ultimas_metricas", None)
        tardo = f" · tardó {m['t_total_s']:.0f} s" if m else ""
        self.lbl_estado.configure(text=f"Versión final lista{tardo}" if lineas else "No se detectó voz",
                                  text_color=C["verde"])

    def _final_error(self, msg):
        self.progreso.grid_forget()
        self.ocupado = False
        self._actualizar_botones()
        if msg == mt.CANCELADO:
            self.lbl_estado.configure(text="Cancelado · sigue el texto en vivo", text_color=C["tenue"])
            return
        self.app.error(f"No se pudo mejorar el texto:\n\n{msg}\n\nLa transcripción en vivo sigue disponible.")

    def abrir_audio(self):
        if self._separando():
            return
        ruta = filedialog.askopenfilename(
            parent=self.app, title="Elige un audio para transcribir",
            initialdir=os.path.join(DATOS, "transcripciones") if os.path.isdir("transcripciones") else DATOS,
            filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.webm *.aac"), ("Todos", "*.*")])
        if not ruta:
            return
        self.lineas = {"vivo": [], "final": []}
        self.ruta_audio = ruta
        self.vista = "final"
        self.redibujar()
        self.correr_pasada_final(ruta, "Transcribiendo " + os.path.basename(ruta))

    # ---------------------------------------------------------------- dibujo
    def _cambiar_vista(self, valor):
        self.vista = "final" if valor == "Final" else "vivo"
        self.redibujar(arriba=self.vista == "final")

    @staticmethod
    def _juntar(lista, linea):
        """Misma persona y pausa corta: se junta en el mismo mensaje. Regresa True si se juntó."""
        anterior = lista[-1] if lista else None
        if anterior and anterior["nombre"] == linea["nombre"] and linea["inicio"] - anterior["fin"] < 1.5:
            anterior["texto"] += " " + linea["texto"]
            anterior["fin"] = linea["fin"]
            anterior["solape"] = anterior["solape"] or linea["solape"]
            return True
        lista.append(dict(linea))
        return False

    def agregar_linea(self, linea, cual):
        abajo = self.lista._parent_canvas.yview()[1] > 0.97
        lista = self.lineas[cual]
        juntada = self._juntar(lista, linea)
        if self.vista == cual:
            if juntada and self.filas:
                self._actualizar_fila(self.filas[-1], lista[-1])
            else:
                self._quitar_vacio()
                self.filas.append(self._fila(lista[-1], len(self.filas)))
            self._acomodar_escuchando()
            if abajo:
                self._al_final()
        self._actualizar_botones()

    def _ancho_texto(self):
        escala = ctk.ScalingTracker.get_widget_scaling(self.hoja)
        return max(self.hoja.winfo_width() / escala - 140, 260)

    def _fila(self, l, i):
        colores = self.app.color(l["nombre"])
        f = ctk.CTkFrame(self.lista, fg_color="transparent", width=1, height=1)
        f.grid(row=i, column=0, sticky="ew", padx=14, pady=(14 if i == 0 else 0, 22))
        f.grid_columnconfigure(1, weight=1)
        Avatar(f, l["nombre"], colores, 36).grid(row=0, column=0, rowspan=2,
                                                                             sticky="n", padx=(0, 14))
        cab = ctk.CTkFrame(f, fg_color="transparent", width=1, height=1)
        cab.grid(row=0, column=1, sticky="w")
        ctk.CTkLabel(cab, text=l["nombre"], font=self.F_("negrita"), text_color=C["texto"], height=20).pack(
            side="left")
        f.tiempo = ctk.CTkLabel(cab, text=mmss(l["inicio"]), font=self.F_("mini"), text_color=C["tenue"], height=20)
        f.tiempo.pack(side="left", padx=10)
        f.etiqueta = ctk.CTkLabel(cab, text="Voces encimadas", font=self.F_("mini"), text_color=C["suave"],
                                  fg_color=C["gris"], corner_radius=10, height=22)
        f.texto = ctk.CTkLabel(f, text=l["texto"], font=self.F_("mensaje"), text_color=C["texto"], anchor="w",
                               justify="left", wraplength=self._ancho_texto())
        f.texto.grid(row=1, column=1, sticky="w", pady=(2, 0))
        self._actualizar_fila(f, l)
        return f

    def _actualizar_fila(self, f, l):
        f.texto.configure(text=l["texto"])
        if l.get("solape"):
            f.etiqueta.pack(side="left", padx=(0, 4), ipadx=6)
        else:
            f.etiqueta.pack_forget()

    def _quitar_vacio(self):
        self.vacio.grid_forget()

    def _acomodar_escuchando(self):
        if self.app.motor.en_curso() and self.vista == "vivo":
            self.escuchando.grid(row=len(self.filas) + 1, column=0, sticky="w", padx=64, pady=(0, 14))
        else:
            self.escuchando.grid_forget()

    def _al_final(self):
        self.update_idletasks()
        self.lista._parent_canvas.yview_moveto(1.0)

    def _al_redimensionar(self, e=None):
        if getattr(self, "_espera_ancho", None):
            self.after_cancel(self._espera_ancho)
        self._espera_ancho = self.after(60, self._reacomodar_texto)

    def _reacomodar_texto(self):
        self._espera_ancho = None
        ancho = int(self._ancho_texto())
        if ancho == getattr(self, "_ultimo_ancho", None):
            return
        self._ultimo_ancho = ancho
        for f in self.filas:
            f.texto.configure(wraplength=ancho)
        self.vacio.configure(wraplength=ancho)

    def redibujar(self, arriba=False):
        for f in self.filas:
            f.destroy()
        self.filas = []
        lineas = self.lineas[self.vista]
        if not lineas:
            if self.ocupado:
                msg = "Preparando la transcripción..."
            elif self.app.motor.en_curso():
                msg = "La transcripción aparece aquí en cuanto alguien hable."
            elif not self.app.listo:
                msg = "Cargando los modelos de voz. La primera vez puede tardar un par de minutos."
            else:
                msg = "Presiona el botón rojo para empezar a grabar,\no abre una grabación que ya tengas."
            self.vacio.configure(text=msg)
            self.vacio.grid(row=0, column=0, pady=150)
        else:
            self._quitar_vacio()
        for i, l in enumerate(lineas):
            self.filas.append(self._fila(l, i))
        self._acomodar_escuchando()
        if self.lineas["final"] and self.lineas["vivo"]:
            self.selector.pack(side="right", padx=(16, 0))
            self.selector.set("Final" if self.vista == "final" else "En vivo")
        else:
            self.selector.pack_forget()
        if arriba:
            self.update_idletasks()
            self.lista._parent_canvas.yview_moveto(0.0)
        elif lineas:
            self._al_final()

    # ---------------------------------------------------------------- exportar
    def _texto_plano(self):
        return "\n".join(mt.formatear(l) for l in self.lineas[self.vista])

    def guardar(self):
        if not self.lineas[self.vista]:
            return
        base = os.path.splitext(os.path.basename(self.ruta_audio or "conversacion"))[0]
        sufijo = "_final" if self.vista == "final" else ""
        ruta = filedialog.asksaveasfilename(
            parent=self.app, title="Guardar transcripción", defaultextension=".txt",
            initialdir=os.path.join(DATOS, "transcripciones") if os.path.isdir("transcripciones") else DATOS,
            initialfile=f"{base}{sufijo}.txt", filetypes=[("Texto", "*.txt")])
        if ruta:
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(self._texto_plano() + "\n")
            self.lbl_estado.configure(text="Guardado", text_color=C["verde"])

    def copiar(self):
        self.app.clipboard_clear()
        self.app.clipboard_append(self._texto_plano())
        self.lbl_estado.configure(text="Copiado", text_color=C["verde"])

    def limpiar(self):
        if self.lineas[self.vista] and not messagebox.askyesno("Limpiar", "¿Borrar la transcripción de la pantalla? "
                                                               "Lo grabado sigue guardado en la carpeta de "
                                                               "transcripciones.", parent=self.app):
            return
        self.lineas = {"vivo": [], "final": []}
        self.vista = "vivo"
        self.lbl_reloj.configure(text="00:00")
        self.redibujar()
        self._actualizar_botones()


# ====================================================================== personas

class PaginaPersonas(Pagina):
    COLUMNAS = 3

    def __init__(self, padre, app):
        super().__init__(padre, app)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        cab = app.encabezado(self, "Personas", "Voces que la app reconoce por su nombre")
        cab.grid(row=0, column=0, sticky="ew", padx=40, pady=(28, 18))
        self.btn_nueva = boton(cab.derecha, "Enrolar persona", lambda: self.app.paginas["enrolar"].nueva(),
                               "principal", font=app.F["boton"], height=44, image=app.I["mas"])
        self.btn_nueva.pack(side="right")
        self.cuerpo = ctk.CTkScrollableFrame(self, fg_color=C["fondo"], corner_radius=0,
                                             scrollbar_button_color=C["barra"])
        self.cuerpo.grid(row=1, column=0, sticky="nsew", padx=(28, 16), pady=(0, 24))
        for c in range(self.COLUMNAS):
            self.cuerpo.grid_columnconfigure(c, weight=1, uniform="tarjetas")
        self.prueba = None
        self.al_cambiar_personas()

    def al_mostrar(self):
        self.al_cambiar_personas()

    def al_cambiar_modelos(self):
        self.al_cambiar_personas()

    def al_cambiar_personas(self):
        for w in self.cuerpo.winfo_children():
            w.destroy()
        self.prueba = None
        self.btn_nueva.configure(state="normal" if self.app.listo else "disabled")
        if not self.app.listo:
            ctk.CTkLabel(self.cuerpo, text="Cargando modelos...", font=self.app.F["sub"],
                         text_color=C["tenue"]).grid(row=0, column=0, columnspan=3, pady=120)
            return
        personas = self.app.motor.personas()
        if not personas:
            vacio = tarjeta(self.cuerpo)
            vacio.grid(row=0, column=0, columnspan=3, sticky="ew", padx=12, pady=12)
            ctk.CTkLabel(vacio, text="Todavía no hay nadie enrolado", font=self.app.F["seccion"],
                         text_color=C["texto"]).pack(pady=(40, 4))
            ctk.CTkLabel(vacio, text="Enrola a cada persona una vez (unos 4 minutos) para que la app "
                                     "la reconozca por su nombre.", font=self.app.F["normal"],
                         text_color=C["suave"]).pack()
            boton(vacio, "Enrolar persona", lambda: self.app.paginas["enrolar"].nueva(), "principal",
                  font=self.app.F["boton"], height=44, image=self.app.I["mas"]).pack(pady=(18, 40))
            return
        nombres_galeria = [p["nombre"] for p in personas]
        for n in nombres_galeria:
            self.app.color(n)
        for i, p in enumerate(personas):
            self._tarjeta_persona(p, i // self.COLUMNAS, i % self.COLUMNAS)
        fila = (len(personas) - 1) // self.COLUMNAS + 1
        self.prueba = tarjeta(self.cuerpo)
        self.prueba_fila = fila
        self._parecidos(fila + 1)
        self._metricas(fila + 2)

    def _tarjeta_persona(self, p, fila, col):
        F, I = self.app.F, self.app.I
        t = tarjeta(self.cuerpo)
        t.grid(row=fila, column=col, sticky="nsew", padx=8, pady=8)
        arriba = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        arriba.pack(fill="x", padx=20, pady=(20, 14))
        Avatar(arriba, p["nombre"], self.app.color(p["nombre"]), 44).pack(side="left")
        bote = ctk.CTkButton(arriba, text="", image=I["basura_suave"], width=34, height=34, corner_radius=8,
                             fg_color="transparent", hover_color=C["rojo_suave"],
                             command=lambda n=p["nombre"]: self.borrar(n))
        bote.pack(side="right", anchor="n")
        bote.bind("<Enter>", lambda e, b=bote: b.configure(image=I["basura"]), add="+")
        bote.bind("<Leave>", lambda e, b=bote: b.configure(image=I["basura_suave"]), add="+")
        textos = ctk.CTkFrame(arriba, fg_color="transparent", width=1, height=1)
        textos.pack(side="left", padx=12)
        ctk.CTkLabel(textos, text=p["nombre"], font=F["seccion"], text_color=C["texto"], height=22,
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(textos, text=f"{p['tomas']} tomas · {max(1, round(p['segundos'] / 60))} min de voz", font=F["chico"],
                     text_color=C["tenue"], height=18, anchor="w").pack(anchor="w")
        if p["en_galeria"]:
            ok, texto = True, "Huella de voz lista"
        else:
            ok, texto = False, "Falta recalcular la huella"
        ctk.CTkLabel(t, text="  " + texto, image=I["check"] if ok else I["aviso"], compound="left",
                     font=F["chico"], text_color=C["texto2"], anchor="w").pack(fill="x", padx=20)
        fila_aqui = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        fila_aqui.pack(fill="x", padx=20, pady=(8, 0))
        ctk.CTkLabel(fila_aqui, text="En la plática", font=F["chico"], text_color=C["texto2"]).pack(side="left")
        var = tk.BooleanVar(value=self.app.motor.presente(p["nombre"]))
        Interruptor(fila_aqui, var, lambda n=p["nombre"], v=var: self._presencia(n, v)).pack(side="right")
        botones = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        botones.pack(fill="x", padx=20, pady=(12, 20))
        boton(botones, "Probar voz", self.probar, "secundario", font=F["chico"], height=40, width=60).pack(
            side="left", fill="x", expand=True)
        boton(botones, "Re-enrolar", lambda n=p["nombre"]: self.app.paginas["enrolar"].nueva(n), "secundario",
              font=F["chico"], height=40, width=60).pack(side="left", fill="x", expand=True, padx=(8, 0))

    def _parecidos(self, fila):
        F = self.app.F
        nombres, M = self.app.motor.parecidos()
        if len(nombres) < 2:
            return
        t = tarjeta(self.cuerpo)
        t.grid(row=fila, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        t.grid_columnconfigure(1, weight=1)
        izq = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        izq.grid(row=0, column=0, sticky="nw", padx=24, pady=24)
        ctk.CTkLabel(izq, text="Parecido entre voces", font=F["seccion"], text_color=C["texto"],
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(izq, text="Entre más bajo, mejor las distingue la app. Si dos voces\nse parecen "
                               "demasiado, conviene repetir el enrolamiento\ncon más variedad de tono.",
                     font=F["chico"], text_color=C["suave"], justify="left", anchor="w").pack(anchor="w", pady=(6, 10))
        pares = [(M[i, j], nombres[i], nombres[j]) for i in range(len(nombres)) for j in range(i + 1, len(nombres))]
        peor = max(pares)
        if peor[0] >= mt.PARECIDO_ALTO:
            icono, texto, color = self.app.I["error"], f"{peor[1]} y {peor[2]} se parecen demasiado", C["rojo"]
        elif peor[0] >= mt.CONSISTENCIA_MINIMA:
            icono, texto, color = self.app.I["aviso"], f"{peor[1]} y {peor[2]} se parecen un poco", C["ambar"]
        else:
            icono, texto, color = self.app.I["check"], "Ningún par se parece demasiado", C["verde"]
        ctk.CTkLabel(izq, text="  " + texto, image=icono, compound="left", font=F["negrita"],
                     text_color=color, anchor="w").pack(anchor="w")
        tabla = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        tabla.grid(row=0, column=1, sticky="e", padx=24, pady=24)
        mostrar = nombres[:6]
        for j, n in enumerate(mostrar):
            ctk.CTkLabel(tabla, text=n, font=F["mini"], text_color=C["tenue"]).grid(row=0, column=j + 1, padx=3,
                                                                                 pady=(0, 4))
        for i, n in enumerate(mostrar):
            ctk.CTkLabel(tabla, text=n, font=F["negrita"], text_color=C["texto"], anchor="w").grid(
                row=i + 1, column=0, sticky="w", padx=(0, 10))
            for j in range(len(mostrar)):
                if i == j:
                    texto, fondo, tinta = "—", C["gris"], C["tenue"]
                else:
                    v = float(M[i, j])
                    texto, tinta = f"{v:.2f}", C["texto"]
                    fondo = (C["rojo_suave"] if v >= mt.PARECIDO_ALTO else
                             C["ambar_suave"] if v >= mt.CONSISTENCIA_MINIMA else C["verde_suave"])
                ctk.CTkLabel(tabla, text=texto, width=86, height=40, corner_radius=10, fg_color=fondo,
                             text_color=tinta, font=F["negrita"] if i != j else F["chico"]).grid(
                    row=i + 1, column=j + 1, padx=3, pady=3)

    def _presencia(self, nombre, var):
        if not self.app.paginas["conversacion"].cambiar_presencia(nombre, bool(var.get())):
            var.set(not var.get())

    # ---------------------------------------------------------------- métricas por persona
    def _metricas(self, fila):
        F = self.app.F
        t = tarjeta(self.cuerpo)
        t.grid(row=fila, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        cab = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        cab.pack(fill="x", padx=24, pady=(22, 0))
        ctk.CTkLabel(cab, text="Métricas por persona", font=F["seccion"], text_color=C["texto"]).pack(side="left")
        self.btn_medir = boton(cab, "Medir", self.medir, "secundario", font=F["chico"], height=38, width=90)
        self.btn_medir.pack(side="right")
        ajustar_texto(ctk.CTkLabel(t, text="Prueba cada toma contra las demás de esa persona. Se guarda en "
                                           "transcripciones/metricas_personas.csv.", font=F["chico"], text_color=C["suave"],
                                   anchor="w", justify="left"), t, 60).pack(anchor="w", fill="x", padx=24, pady=(4, 18))
        self.tabla_metricas = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        self.tabla_metricas.pack(fill="x", padx=24, pady=(0, 20))
        if getattr(self, "datos_metricas", None):
            self._pintar_metricas(self.datos_metricas)

    def medir(self):
        self.btn_medir.configure(state="disabled", text="Midiendo...")
        self.app.en_hilo(self.app.motor.metricas_personas, self._medido, self._fallo_medir)

    def _fallo_medir(self, msg):
        self.btn_medir.configure(state="normal", text="Medir")
        self.app.error(f"No se pudo medir:\n\n{msg}")

    def _medido(self, filas):
        self.datos_metricas = filas
        if self.btn_medir.winfo_exists():
            self.btn_medir.configure(state="normal", text="Medir")
            self._pintar_metricas(filas)

    def _pintar_metricas(self, filas):
        F = self.app.F
        for w in self.tabla_metricas.winfo_children():
            w.destroy()
        for c in range(6):
            self.tabla_metricas.grid_columnconfigure(c, weight=1)
        titulos = ("Persona", "Parecido consigo\n(bajo / típico / alto)", "Umbral", "Más parecido a", "Aciertos\ncon 1 s", "Aciertos\ncon 3 s")
        for c, x in enumerate(titulos):
            ctk.CTkLabel(self.tabla_metricas, text=x, font=F["mini"], text_color=C["tenue"], justify="left", anchor="w").grid(
                row=0, column=c, sticky="w", padx=(0, 10), pady=(0, 6))
        for i, f in enumerate(filas, 1):
            parecido = f"{f['bajo']} / {f['tipico']} / {f['alto']}" if f["tipico"] != "" else "falta más de una toma"
            otro = f"{f['mas_parecido']} ({f['parecido']})" if f["mas_parecido"] else "—"
            acierta = [f"{f[k]} %" if f[k] != "" else "—" for k in ("aciertos_1s", "aciertos_3s")]
            for c, x in enumerate((f["nombre"], parecido, f"{f['umbral']:.2f}", otro, acierta[0], acierta[1])):
                ctk.CTkLabel(self.tabla_metricas, text=x, font=F["negrita"] if c == 0 else F["chico"], text_color=C["texto"],
                             anchor="w").grid(row=i, column=c, sticky="w", padx=(0, 10), pady=2)

    def borrar(self, nombre):
        if not messagebox.askyesno("Borrar persona", f"¿Borrar la voz de {nombre}? Tendrá que volver a enrolarse.",
                                   parent=self.app):
            return
        self.app.en_hilo(lambda: self.app.motor.borrar_persona(nombre), lambda r: self.app.personas_cambiaron())

    # ---------------------------------------------------------------- probar una voz
    def probar(self):
        if self.prueba is None:
            return
        for w in self.prueba.winfo_children():
            w.destroy()
        self.prueba.grid(row=self.prueba_fila, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        fila = ctk.CTkFrame(self.prueba, fg_color="transparent", width=1, height=1)
        fila.pack(fill="x", padx=24, pady=20)
        self.lbl_prueba = ctk.CTkLabel(fila, text="Habla unos 5 segundos...", font=self.app.F["seccion"],
                                       text_color=C["rojo"], anchor="w")
        self.lbl_prueba.pack(side="left")
        medidor = Medidor(fila, barras=14)
        medidor.pack(side="left", padx=20)
        self.barras_prueba = ctk.CTkFrame(self.prueba, fg_color="transparent", width=1, height=1)
        self.barras_prueba.pack(fill="x", padx=24, pady=(0, 20))
        mic = self.app.microfono_elegido()

        def trabajo():
            audio = self.app.motor.grabar_toma(5, mic, lambda r: self.app.post(medidor.poner, r, True))
            self.app.post(self.lbl_prueba.configure, text="Analizando...", text_color=C["tenue"])
            return self.app.motor.identificar(audio)
        self.app.en_hilo(trabajo, self._resultado, self._fallo)

    def _fallo(self, msg):
        if self.prueba is not None:
            self.prueba.grid_forget()
        self.app.error(msg)

    def _resultado(self, r):
        sims, niveles = r
        mejor, nombre = sims[0]
        umbral = niveles.get(nombre, 0.4)
        segundo = sims[1][0] if len(sims) > 1 else -1
        if mejor >= umbral and mejor - segundo >= 0.05:
            self.lbl_prueba.configure(text=f"Suena a {nombre}", text_color=C["verde"])
        elif mejor >= umbral:
            self.lbl_prueba.configure(text=f"¿{nombre}? No está muy claro", text_color=C["ambar"])
        else:
            self.lbl_prueba.configure(text="No se parece a nadie registrado", text_color=C["suave"])
        for s, n in sims[:5]:
            u = niveles.get(n, 0.4)
            f = ctk.CTkFrame(self.barras_prueba, fg_color="transparent", width=1, height=1)
            f.pack(fill="x", pady=3)
            ctk.CTkLabel(f, text=n, width=110, anchor="w", font=self.app.F["chico"], text_color=C["texto"]).pack(
                side="left")
            barra = ctk.CTkProgressBar(f, width=260, height=8, corner_radius=4, fg_color=C["barra"],
                                       progress_color=C["verde"] if s >= u else C["apagado"])
            barra.set(float(np.clip(s, 0, 1)))
            barra.pack(side="left", padx=8)
            ctk.CTkLabel(f, text=f"{s:.2f}  (umbral {u:.2f})", font=self.app.F["mini"],
                         text_color=C["tenue"]).pack(side="left")


# ====================================================================== ajustes

class PaginaAjustes(Pagina):
    def __init__(self, padre, app):
        super().__init__(padre, app)
        F = app.F
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        cab = app.encabezado(self, "Ajustes", "Se guardan solos")
        cab.grid(row=0, column=0, sticky="ew", padx=40, pady=(28, 18))
        cuerpo = ctk.CTkScrollableFrame(self, fg_color=C["fondo"], corner_radius=0, scrollbar_button_color=C["barra"])
        cuerpo.grid(row=1, column=0, sticky="nsew", padx=(32, 16), pady=(0, 24))
        cuerpo.grid_columnconfigure(0, weight=1)
        self.cuerpo = cuerpo

        # --- entorno
        t = self._seccion(0, "¿Dónde vas a grabar?",
                          "La app ajusta sola qué tan sensible es al ruido y a los cambios de persona.")
        rejilla = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        rejilla.pack(fill="x", padx=19, pady=(4, 22))
        self.opciones = {}
        iconos = {"silencioso": "casa", "salon": "salon", "exterior": "sol"}
        for c, (clave, p) in enumerate(mt.PERFILES.items()):
            rejilla.grid_columnconfigure(c, weight=1, uniform="perfil")
            b = OpcionPerfil(rejilla, p["nombre"], p["detalle"], _icono(iconos[clave], C["suave"], 20), F,
                             lambda k=clave: self.elegir_perfil(k))
            b.grid(row=0, column=c, sticky="nsew", padx=5)   # mismo margen en las tres para que midan igual
            self.opciones[clave] = b

        # --- qué pasa al terminar
        t = self._seccion(1, "Al terminar de grabar", "Entre más completo, mejor el resultado pero tarda más.")
        rejilla = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        rejilla.pack(fill="x", padx=19, pady=(4, 22))
        self.modos = {}
        iconos = {"rapido": "sol", "mejorado": "check", "completo": "personas"}
        for c, (clave, m) in enumerate(mt.MODOS.items()):
            rejilla.grid_columnconfigure(c, weight=1, uniform="modo")
            b = OpcionPerfil(rejilla, m["nombre"], m["detalle"], _icono(iconos[clave], C["suave"], 20), F,
                             lambda k=clave: self.elegir_modo(k))
            b.grid(row=0, column=c, sticky="nsew", padx=5)
            self.modos[clave] = b

        # --- inglés
        t = self._seccion(2, None, None)
        fila = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        fila.pack(fill="x", padx=24, pady=22)
        textos = ctk.CTkFrame(fila, fg_color="transparent", width=1, height=1)
        textos.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(textos, text="Traducir palabras en inglés", font=F["seccion"], text_color=C["texto"],
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(textos, text='Los términos técnicos como "deploy" o "backend" siempre se quedan en inglés.',
                     font=F["normal"], text_color=C["suave"], anchor="w").pack(anchor="w", pady=(4, 0))
        self.var_traducir = tk.BooleanVar(value=bool(app.motor.ajustes.get("traducir")))
        self.sw_traducir = Interruptor(fila, self.var_traducir, self._traducir)
        self.sw_traducir.pack(side="right")
        self.lbl_traducir = ctk.CTkLabel(fila, text="", font=F["chico"], text_color=C["suave"])
        self.lbl_traducir.pack(side="right", padx=12)

        # --- equipo
        t = self._seccion(3, None, None)
        fila = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        fila.pack(fill="x", padx=24, pady=22)
        textos = ctk.CTkFrame(fila, fg_color="transparent", width=1, height=1)
        textos.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(textos, text="Equipo para transcribir", font=F["seccion"], text_color=C["texto"],
                     anchor="w").pack(anchor="w")
        ctk.CTkLabel(textos, text="Automático: tarjeta NVIDIA si hay; si no, el procesador.", font=F["normal"],
                     text_color=C["suave"], anchor="w").pack(anchor="w", pady=(4, 0))
        self.chip_equipo = ctk.CTkLabel(fila, text="Cargando...", font=F["chico"], text_color="#1E5E3E",
                                        fg_color=C["verde_suave"], corner_radius=14, height=32)
        self.chip_equipo.pack(side="right", ipadx=10)

        # --- tu equipo: especificaciones y paquete para soporte
        t = self._seccion(4, None, None)
        cab = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        cab.pack(fill="x", padx=24, pady=(22, 4))
        ctk.CTkLabel(cab, text="Tu equipo", font=F["seccion"], text_color=C["texto"]).pack(side="left")
        boton(cab, "Crear paquete", self._paquete, "principal", font=F["chico"], height=38,
              image=app.I["guardar_blanco"]).pack(side="right")
        boton(cab, "Copiar datos", self._copiar_datos, "secundario", font=F["chico"], height=38,
              image=app.I["copiar"]).pack(side="right", padx=8)
        ctk.CTkLabel(t, text="Para mandar los resultados de una prueba: crea el paquete y envía el .zip.",
                     font=F["normal"], text_color=C["suave"], anchor="w").pack(anchor="w", padx=24)
        self.tabla_equipo = ctk.CTkFrame(t, fg_color=C["fondo"], corner_radius=12)
        self.tabla_equipo.pack(fill="x", padx=24, pady=(14, 0))
        self.tabla_equipo.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.tabla_equipo, text="Revisando el equipo...", font=F["chico"],
                     text_color=C["tenue"]).grid(row=0, column=0, padx=16, pady=12, sticky="w")
        fila = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        fila.pack(fill="x", padx=24, pady=(14, 22))
        ctk.CTkLabel(fila, text="Incluir el audio de la última grabación en el paquete", font=F["normal"],
                     text_color=C["texto"]).pack(side="left")
        self.var_audio = tk.BooleanVar(value=True)
        Interruptor(fila, self.var_audio).pack(side="right")
        self._datos_equipo = None

        # --- métricas de la última grabación
        t = self._seccion(5, None, None)
        cab = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        cab.pack(fill="x", padx=24, pady=(22, 4))
        ctk.CTkLabel(cab, text="Métricas de la última grabación", font=F["seccion"], text_color=C["texto"]).pack(side="left")
        boton(cab, "Abrir carpeta", self._abrir_carpeta, "secundario", font=F["chico"], height=38).pack(side="right")
        self.lbl_metricas = ctk.CTkLabel(t, text="", font=F["chico"], text_color=C["suave"], anchor="w", justify="left")
        self.lbl_metricas.pack(fill="x", padx=24, pady=(6, 22))
        ajustar_texto(self.lbl_metricas, t, 60)

        # --- avanzado (plegado)
        self.btn_avanzado = ctk.CTkButton(cuerpo, text="Avanzado", anchor="w", height=56, corner_radius=16,
                                          border_width=1, border_color=C["borde"], fg_color=C["tarjeta"],
                                          hover_color=C["gris"], text_color=C["texto"], font=F["seccion"],
                                          command=self._plegar)
        self.btn_avanzado.grid(row=6, column=0, sticky="ew", padx=8, pady=8)
        self.avanzado = tarjeta(cuerpo)
        self.abierto = False
        self._armar_avanzado()
        self._pintar_perfiles()
        self._pintar_traducir()

    def _seccion(self, fila, titulo, texto):
        t = tarjeta(self.cuerpo)
        t.grid(row=fila, column=0, sticky="ew", padx=8, pady=8)
        if titulo:
            ctk.CTkLabel(t, text=titulo, font=self.app.F["seccion"], text_color=C["texto"], anchor="w").pack(
                anchor="w", padx=24, pady=(22, 0))
            ctk.CTkLabel(t, text=texto, font=self.app.F["normal"], text_color=C["suave"], anchor="w").pack(
                anchor="w", padx=24, pady=(4, 12))
        return t

    def _armar_avanzado(self):
        F = self.app.F
        a = self.avanzado
        self.var_graves = tk.BooleanVar(value=bool(self.app.motor.ajustes.get("filtro_graves", True)))
        self.var_espanglish = tk.BooleanVar(value=bool(self.app.motor.ajustes.get("espanglish", True)))
        self.var_grande = tk.BooleanVar(value=bool(self.app.motor.ajustes.get("final_grande", True)))
        for texto, var, clave in (("Quitar los sonidos muy graves del micrófono (golpes, ventilador)", self.var_graves, "filtro_graves"),
                                  ("Escribir los términos técnicos en inglés", self.var_espanglish, "espanglish"),
                                  ("Sin tarjeta NVIDIA: versión final con el modelo grande (tarda más)", self.var_grande,
                                   "final_grande")):
            fila = ctk.CTkFrame(a, fg_color="transparent", width=1, height=1)
            fila.pack(fill="x", padx=24, pady=(16, 0))
            ctk.CTkLabel(fila, text=texto, font=F["normal"], text_color=C["texto"]).pack(side="left")
            Interruptor(fila, var, lambda v=var, k=clave: self.app.motor.cambiar_ajuste(k, bool(v.get()))).pack(
                side="right")
        fila = ctk.CTkFrame(a, fg_color="transparent", width=1, height=1)
        fila.pack(fill="x", padx=24, pady=(16, 0))
        ctk.CTkLabel(fila, text="Micrófono", font=F["normal"], text_color=C["texto"]).pack(side="left")
        self.combo_mic = ctk.CTkOptionMenu(fila, values=["(cargando)"], width=320, height=36, corner_radius=10,
                                           fg_color=C["gris"], button_color=C["gris_hover"],
                                           button_hover_color=C["barra"], text_color=C["texto"], font=F["chico"],
                                           dropdown_font=F["chico"], command=self._elegir_mic)
        self.combo_mic.pack(side="right")
        fila = ctk.CTkFrame(a, fg_color="transparent", width=1, height=1)
        fila.pack(fill="x", padx=24, pady=(10, 0))
        self.btn_mic = boton(fila, "Probar micrófono (10 s)", self.probar_mic, "secundario", font=F["chico"], height=38)
        self.btn_mic.pack(side="left")
        self.medidor_mic = Medidor(fila, barras=14)
        self.medidor_mic.pack(side="left", padx=12)
        self.lbl_mic = ctk.CTkLabel(a, text="Habla normal durante la prueba, a la distancia a la que vas a grabar.", font=F["chico"],
                                    text_color=C["suave"], anchor="w", justify="left")
        self.lbl_mic.pack(fill="x", padx=24, pady=(6, 0))
        ajustar_texto(self.lbl_mic, a, 60)
        self.lbl_umbrales = ctk.CTkLabel(a, text="", font=F["chico"], text_color=C["suave"], anchor="w",
                                         justify="left")
        self.lbl_umbrales.pack(fill="x", padx=24, pady=(16, 0))
        fila = ctk.CTkFrame(a, fg_color="transparent", width=1, height=1)
        fila.pack(fill="x", padx=24, pady=(16, 22))
        boton(fila, "Abrir carpeta de transcripciones", self._abrir_carpeta, "secundario", font=F["chico"],
              height=38).pack(side="left")
        ctk.CTkLabel(fila, text=f"Versión {mt.tr.VERSION}", font=F["mini"], text_color=C["tenue"]).pack(side="right")

    def al_mostrar(self):
        self._pintar_perfiles()
        self._pintar_modos()
        self._pintar_metricas()
        self._revisar_equipo()
        if self.app.listo:
            self.chip_equipo.configure(text="  Usando: " + self.app.motor.equipo().lower() + "  ")
            umbrales = self.app.motor.umbrales()
            p = self.app.motor.perfil()
            texto = (f"Con «{p['nombre']}»: penalización por cambio de persona {p['penal']:.2f}, "
                     f"detector de voz {p['vad']:.2f}, pausa que cierra una frase {p['silencio_ms']} ms.")
            if umbrales:
                texto += "\nUmbral de cada persona: " + ", ".join(f"{n} {u:.2f}" for n, u in umbrales.items())
            self.lbl_umbrales.configure(text=texto)

    def al_cambiar_modelos(self):
        self._datos_equipo = None
        lista, elegido = self.app.microfonos()
        nombres = [n for _, n in lista] or ["(sin micrófonos)"]
        self.combo_mic.configure(values=nombres)
        actual = next((n for i, n in lista if i == elegido), nombres[0])
        self.combo_mic.set(actual)
        self.al_mostrar()

    def al_cambiar_personas(self):
        self.al_mostrar()

    def _elegir_mic(self, nombre):
        self.app.motor.cambiar_ajuste("microfono", nombre)

    def _plegar(self):
        self.abierto = not self.abierto
        if self.abierto:
            self.avanzado.grid(row=7, column=0, sticky="ew", padx=8, pady=(0, 8))
            self.al_mostrar()
        else:
            self.avanzado.grid_forget()

    def elegir_modo(self, clave):
        self.app.motor.cambiar_ajuste("modo_final", clave)
        self._pintar_modos()

    def _pintar_modos(self):
        motor = self.app.motor
        for clave, b in self.modos.items():
            b.pintar(clave == motor.ajustes["modo_final"])
            t = motor.estimar(120, clave)
            if clave == "rapido":
                extra = "Sin espera"
            elif t is None:
                extra = "El tiempo se mide la primera vez"
            else:
                extra = f"2 min de audio ≈ {max(1, round(t / 60))} min"
            b.lbl_detalle.configure(text=f"{mt.MODOS[clave]['detalle']}\n{extra}")

    def _pintar_metricas(self):
        m = getattr(self.app.motor, "ultimas_metricas", None)
        if not m:
            self.lbl_metricas.configure(text="Todavía no hay datos. Cada grabación se agrega a transcripciones/metricas.csv.")
            return
        texto = (f"{m['audio']} · {m['duracion_s']} s · modo {m['modo']}\n"
                 f"Tardó {m['t_total_s']} s ({m['veces_lo_que_dura']} veces lo que dura el audio): quién habla {m['t_quien_habla_s']} s, "
                 f"transcribir {m['t_transcribir_s']} s, separar voces {m['t_separar_voces_s']} s\n"
                 f"Encimados: {m['encimados_revisados']} revisados, {m['encimados_separados']} separados · "
                 f"dudoso {m['pct_dudoso']} % · desconocido {m['pct_desconocido']} %\n"
                 f"Tiempo de habla: {m['habla_por_persona'] or '—'}")
        if m["atraso_vivo_max_s"] != "":
            texto += f"\nAtraso en vivo: promedio {m['atraso_vivo_prom_s']} s, máximo {m['atraso_vivo_max_s']} s"
        self.lbl_metricas.configure(text=texto)

    def probar_mic(self):
        conv = self.app.paginas["conversacion"]
        if self.app.motor.en_curso() or conv.ocupado or self.app.paginas["separar"].ocupado:
            messagebox.showinfo("Transcriptor", "Espera a que termine lo que está en curso.", parent=self.app)
            return
        self.btn_mic.configure(state="disabled", text="Escuchando...")
        self.lbl_mic.configure(text="Habla normal, a la distancia a la que vas a grabar...", text_color=C["acento_hover"])
        self.medidor_mic.reiniciar()
        mic = self.app.microfono_elegido()
        self.app.en_hilo(lambda: self.app.motor.probar_microfono(mic, lambda r: self.app.post(self.medidor_mic.poner, r, True)),
                         self._mic_listo, self._mic_error)

    def _mic_error(self, msg):
        self.btn_mic.configure(state="normal", text="Probar micrófono (10 s)")
        self.lbl_mic.configure(text=f"No se pudo probar el micrófono: {msg}", text_color=C["rojo"])

    def _mic_listo(self, r):
        _, info = r
        self.btn_mic.configure(state="normal", text="Probar micrófono (10 s)")
        ruido = f"{info['ruido_db']:.0f} dB" if info["ruido_db"] is not None else "sin medir"
        texto = f"Voz {info['voz_db']:.0f} dB · ruido {ruido} · {info['voz_seg']:.0f} s de voz."
        if info["avisos"]:
            texto += " " + " ".join(info["avisos"])
        else:
            texto += " Se oye bien."
        self.lbl_mic.configure(text=texto, text_color=C["verde"] if info["calidad"] == "buena" else C["ambar"])

    def elegir_perfil(self, clave):
        if self.app.motor.en_curso():
            messagebox.showinfo("Ajustes", "El cambio se aplica en la siguiente grabación.", parent=self.app)
        self.app.motor.cambiar_ajuste("perfil", clave)
        self.app.entorno_cambio()
        self.al_mostrar()

    def _pintar_perfiles(self):
        actual = self.app.motor.ajustes["perfil"]
        for clave, b in self.opciones.items():
            b.pintar(clave == actual)

    def _pintar_traducir(self, texto=None):
        if texto is None:
            texto = "Activado" if self.var_traducir.get() else "Desactivado"
        self.lbl_traducir.configure(text=texto)

    def _traducir(self):
        quiere = bool(self.var_traducir.get())
        self.app.motor.cambiar_ajuste("traducir", quiere)
        if not quiere:
            self._pintar_traducir()
            return
        self.sw_traducir.habilitar(False)
        self._pintar_traducir("Cargando el traductor...")

        def listo(_):
            self.sw_traducir.habilitar(True)
            self._pintar_traducir()

        def fallo(msg):
            self.var_traducir.set(False)
            self.app.motor.cambiar_ajuste("traducir", False)
            self.sw_traducir.habilitar(True)
            self._pintar_traducir()
            self.app.error(msg)
        self.app.en_hilo(self.app.motor.preparar_traductor, listo, fallo)

    # ---------------------------------------------------------------- tu equipo
    def _revisar_equipo(self):
        if self._datos_equipo is not None:
            return
        self._datos_equipo = {}
        motor = self.app.motor if self.app.listo else None
        self.app.en_hilo(lambda: diagnostico.resumen(motor), self._pintar_equipo, lambda m: None)

    def _pintar_equipo(self, datos):
        self._datos_equipo = datos
        for w in self.tabla_equipo.winfo_children():
            w.destroy()
        for i, (k, v) in enumerate(datos.items()):
            ctk.CTkLabel(self.tabla_equipo, text=k, font=self.app.F["chico"], text_color=C["tenue"], anchor="w").grid(
                row=i, column=0, sticky="nw", padx=(16, 20), pady=(10 if i == 0 else 2, 10 if i == len(datos) - 1 else 2))
            valor = ctk.CTkLabel(self.tabla_equipo, text=v, font=self.app.F["chico"], text_color=C["texto"],
                                 anchor="w", justify="left")
            valor.grid(row=i, column=1, sticky="w", padx=(0, 16),
                       pady=(10 if i == 0 else 2, 10 if i == len(datos) - 1 else 2))
            ajustar_texto(valor, self.tabla_equipo, 190)

    def _copiar_datos(self):
        motor = self.app.motor if self.app.listo else None

        def listo(texto):
            self.app.clipboard_clear()
            self.app.clipboard_append(texto)
            messagebox.showinfo("Tu equipo", "Datos copiados. Ya los puedes pegar en el chat o en un mensaje.",
                                parent=self.app)
        self.app.en_hilo(lambda: diagnostico.texto_completo(motor, mt.tr.VERSION), listo)

    def _paquete(self):
        sello = time.strftime("%Y%m%d_%H%M")
        ruta = filedialog.asksaveasfilename(parent=self.app, title="Guardar paquete para soporte",
                                            defaultextension=".zip", initialdir=os.path.expanduser("~"),
                                            initialfile=f"transcriptor_paquete_{sello}.zip",
                                            filetypes=[("Archivo zip", "*.zip")])
        if not ruta:
            return
        motor = self.app.motor if self.app.listo else None
        audio = bool(self.var_audio.get())

        def listo(r):
            messagebox.showinfo("Tu equipo", f"Listo, se guardó:\n{r}\n\nMándalo tal cual. No lleva las voces "
                                             "enroladas de nadie.", parent=self.app)
        self.app.en_hilo(lambda: diagnostico.crear_paquete(ruta, motor, mt.tr.VERSION, audio), listo)

    def _abrir_carpeta(self):
        carpeta = os.path.join(DATOS, mt.tr.CARPETA_SALIDA)
        os.makedirs(carpeta, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(carpeta)
            else:
                import subprocess
                subprocess.Popen(["xdg-open", carpeta])
        except Exception as e:
            self.app.error(f"No se pudo abrir la carpeta:\n{carpeta}\n\n{e}")


# ====================================================================== enrolamiento guiado

class PaginaEnrolar(Pagina):
    def __init__(self, padre, app):
        super().__init__(padre, app)
        self.pasos = app.motor.pasos
        self.grabando = False
        self.tomas, self.info = {}, {}
        self.nombre, self.indice = None, 0
        self.detener_evt = None
        self.reemplazar = None
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        arriba = ctk.CTkFrame(self, fg_color="transparent", width=1, height=1)
        arriba.grid(row=0, column=0, sticky="ew", padx=40, pady=(24, 0))
        self.btn_volver = ctk.CTkButton(arriba, text="Volver a Personas", image=app.I["atras_suave"],
                                        compound="left", anchor="w", fg_color="transparent", hover_color=C["gris"],
                                        text_color=C["suave"], font=app.F["chico"], height=44, width=170,
                                        command=self.volver)
        self.btn_volver.pack(side="left")
        self.lbl_quien = ctk.CTkLabel(arriba, text="", font=app.F["chico"], text_color=C["tenue"])
        self.lbl_quien.pack(side="right")
        self.marco = ctk.CTkScrollableFrame(self, fg_color=C["fondo"], corner_radius=0,
                                            scrollbar_button_color=C["barra"])
        self.marco.grid(row=1, column=0, sticky="nsew", padx=(40, 16), pady=(8, 24))
        self.area = ctk.CTkFrame(self.marco, fg_color="transparent", width=1, height=1)
        self.area.pack(fill="both", expand=True)
        # botones de grabar y el resultado de la toma van fijos abajo: así siempre se ven,
        # aunque la pantalla sea chica y la frase no quepa completa
        self.abajo = ctk.CTkFrame(self, fg_color="transparent", width=1, height=1)
        self.vista_inicio()

    def F_(self, k):
        return self.app.F[k]

    def limpiar(self):
        for w in self.area.winfo_children() + self.abajo.winfo_children():
            w.destroy()
        self.abajo.grid_forget()
        self.marco.grid(pady=(8, 24))

    def chica(self):
        """¿La pantalla es baja? (laptops con escala de 125-150 %). Entonces se aprieta el espacio."""
        escala = ctk.ScalingTracker.get_widget_scaling(self)
        return self.winfo_height() / escala < 700

    def centro(self, ancho=780):
        """Columna centrada de ancho máximo `ancho` (como el max-width de la propuesta)."""
        c = ctk.CTkFrame(self.area, fg_color="transparent", width=1, height=1)
        c.pack(fill="both", expand=True, pady=10)
        self.after(10, lambda: self.marco._parent_canvas.yview_moveto(0.0))

        def acomodar(e=None):
            if not c.winfo_exists():
                return
            escala = ctk.ScalingTracker.get_widget_scaling(self.area)
            libre = int(max(self.area.winfo_width() / escala - ancho, 0) // 2)
            c.pack_configure(padx=libre)
        self.area.bind("<Configure>", acomodar)
        self.after(10, acomodar)
        return c

    def volver(self):
        if self.grabando:
            return
        if self.tomas and not messagebox.askyesno("Enrolar", "¿Salir y descartar las tomas grabadas?",
                                                  parent=self.app):
            return
        self.tomas, self.info = {}, {}
        self.app.mostrar("personas")
        self.vista_inicio()

    # ---------------------------------------------------------------- inicio
    def nueva(self, nombre=None):
        if self.grabando:
            return
        self.app.mostrar("enrolar")
        self.vista_inicio(nombre)

    def vista_inicio(self, nombre=None):
        self.limpiar()
        self.tomas, self.info = {}, {}
        self.lbl_quien.configure(text="")
        c = self.centro(620)
        t = tarjeta(c, corner_radius=20)
        t.pack(fill="x", pady=(40, 0))
        ctk.CTkLabel(t, text="¿Quién se va a enrolar?", font=self.F_("grande"), text_color=C["texto"]).pack(
            anchor="w", padx=36, pady=(34, 4))
        ctk.CTkLabel(t, text="Vas a leer 8 frases cortas con distintos tonos y platicar unos segundos.\n"
                             "Son unos 4 minutos. Busca un lugar callado.", font=self.F_("normal"),
                     text_color=C["suave"], justify="left").pack(anchor="w", padx=36)
        self.entrada = ctk.CTkEntry(t, placeholder_text="Nombre", height=48, corner_radius=12, font=self.F_("grande"),
                                    border_color=C["borde"], fg_color=C["fondo"], text_color=C["texto"])
        self.entrada.pack(fill="x", padx=36, pady=(22, 8))
        if nombre:
            self.entrada.insert(0, nombre)
            ctk.CTkLabel(t, text=f"Se reemplazarán las tomas anteriores de {nombre}.", font=self.F_("chico"),
                         text_color=C["ambar"]).pack(anchor="w", padx=36)
        self.entrada.bind("<Return>", lambda e: self.comenzar())
        entrada = self.entrada
        self.after(100, lambda: entrada.winfo_exists() and entrada.focus_set())
        self.btn_comenzar = boton(t, "Comenzar", self.comenzar, "principal", font=self.F_("boton"), height=48)
        self.btn_comenzar.pack(fill="x", padx=36, pady=(16, 34))
        self.btn_comenzar.configure(state="normal" if self.app.listo else "disabled")

    def al_cambiar_modelos(self):
        if hasattr(self, "btn_comenzar") and self.btn_comenzar.winfo_exists():
            self.btn_comenzar.configure(state="normal")

    def comenzar(self):
        if not self.app.listo:
            return
        nombre, err = mt.nombre_valido(self.entrada.get())
        if err:
            messagebox.showwarning("Enrolar", err, parent=self.app)
            return
        existentes = [p["nombre"].lower() for p in self.app.motor.personas()]
        if nombre.lower() in existentes and not messagebox.askyesno(
                "Enrolar", f"{nombre} ya está enrolado. ¿Reemplazar su voz con este nuevo enrolamiento?",
                parent=self.app):
            return
        self.nombre, self.mic_id = nombre, self.app.microfono_elegido()
        self.tomas, self.info = {}, {}
        self.ir_a(0)

    # ---------------------------------------------------------------- pasos
    def _progreso(self, padre):
        cab = ctk.CTkFrame(padre, fg_color="transparent", width=1, height=1)
        cab.pack(fill="x")
        ctk.CTkLabel(cab, text=f"Frase {self.indice + 1} de {len(self.pasos)} · {self.pasos[self.indice]['titulo']}",
                     font=self.F_("chico"), text_color=C["tenue"]).pack(side="left")
        hechas = sum(1 for p in self.pasos if p["id"] in self.info)
        ctk.CTkLabel(cab, text=f"{hechas} de {len(self.pasos)} listas", font=self.F_("chico"),
                     text_color=C["tenue"]).pack(side="right")
        barras = ctk.CTkFrame(padre, fg_color="transparent", width=1, height=1)
        barras.pack(fill="x", pady=(10, 0))
        for i, p in enumerate(self.pasos):
            barras.grid_columnconfigure(i, weight=1, uniform="pasos")
            info = self.info.get(p["id"])
            if i == self.indice:
                color = C["acento_borde"]
            elif info:
                color = {"buena": C["acento"], "regular": "#E0A800", "mala": C["rojo"]}[info["calidad"]]
            else:
                color = C["barra"]
            b = ctk.CTkButton(barras, text="", height=6, corner_radius=3, fg_color=color, hover_color=color,
                              command=lambda k=i: not self.grabando and self.ir_a(k))
            b.grid(row=0, column=i, sticky="ew", padx=3)

    def ir_a(self, i):
        self.indice = i
        p = self.pasos[i]
        self.limpiar()
        self.lbl_quien.configure(text=f"Enrolando a {self.nombre}")
        c = self.centro()
        chica = self.chica()
        self._progreso(c)
        t = tarjeta(c, corner_radius=20)
        t.pack(fill="x", pady=(16 if chica else 26, 0))
        pastilla = ctk.CTkLabel(t, text=p["instruccion"], font=self.F_("chico"), text_color=C["acento_hover"],
                                fg_color=C["acento_suave"], corner_radius=14, height=30)
        pastilla.pack(pady=(20 if chica else 34, 0), padx=30, ipadx=12)
        ajustar_texto(pastilla, t, 110, 640)
        libre = p.get("libre")
        texto = p["texto"] if libre else f"“{p['texto']}”"
        if libre:
            fuente = self.F_("seccion")
        elif len(texto) < 150 and not chica:
            fuente = self.F_("lectura")
        else:
            fuente = self.F_("lectura_larga")
        lectura = ctk.CTkLabel(t, text=texto, font=fuente, text_color=C["texto"] if not libre else C["suave"],
                               justify="center")
        lectura.pack(padx=40, pady=(14, 6) if chica else (22, 10))
        ajustar_texto(lectura, t, 100, 640)
        self.medidor = Medidor(t, barras=16, alto=30 if chica else 40, ancho_barra=5, separacion=4)
        self.medidor.pack(pady=(4 if chica else 6, 0))
        self.lbl_cuenta = ctk.CTkLabel(t, text=f"Hasta {p['segundos']} segundos", font=self.F_("chico"),
                                       text_color=C["suave"])
        self.lbl_cuenta.pack(pady=(4, 18) if chica else (8, 30))

        self.marco.grid(pady=(8, 0))
        self.abajo.grid(row=2, column=0, sticky="ew", padx=(40, 16), pady=(10 if chica else 14, 18 if chica else 24))
        controles = ctk.CTkFrame(self.abajo, fg_color="transparent", width=1, height=1)
        controles.pack()
        alto = 44 if chica else 48
        self.btn_ant = boton(controles, "Anterior", lambda: self.ir_a(self.indice - 1), "secundario",
                             font=self.F_("boton_normal"), height=alto, width=130)
        self.btn_ant.pack(side="left", padx=7)
        self.btn_escuchar = boton(controles, "Escuchar", self.escuchar, "secundario", font=self.F_("boton_normal"),
                                  height=alto, width=130)
        self.btn_escuchar.pack(side="left", padx=7)
        self.btn_toma = BotonRedondo(controles, 56 if chica else 64, self.alternar_toma)
        self.btn_toma.pack(side="left", padx=14)
        ultimo = i == len(self.pasos) - 1
        self.btn_sig = boton(controles, "Revisar" if ultimo else "Siguiente", self.siguiente, "principal",
                             font=self.F_("boton"), height=alto, width=130)
        self.btn_sig.pack(side="left", padx=7)
        self.resultado = ctk.CTkFrame(self.abajo, fg_color="transparent", width=1, height=1)
        self.resultado.pack(fill="x", pady=(10, 0))
        self._mostrar_resultado()

    def _mostrar_resultado(self):
        for w in self.resultado.winfo_children():
            w.destroy()
        p = self.pasos[self.indice]
        info = self.info.get(p["id"])
        self.btn_ant.configure(state="normal" if self.indice > 0 and not self.grabando else "disabled")
        self.btn_sig.configure(state="normal" if info and not self.grabando else "disabled")
        self.btn_escuchar.configure(state="normal" if p["id"] in self.tomas and not self.grabando else "disabled")
        if not info:
            return
        titulo = {"buena": "Toma buena", "regular": "Toma aceptable", "mala": "Hay que repetir esta toma"}
        icono = {"buena": self.app.I["check"], "regular": self.app.I["aviso"], "mala": self.app.I["error"]}
        ctk.CTkLabel(self.resultado, text="  " + titulo[info["calidad"]], image=icono[info["calidad"]],
                     compound="left", font=self.F_("negrita"), text_color=C["texto"]).pack()
        for a in info["avisos"]:
            ajustar_texto(ctk.CTkLabel(self.resultado, text=a, font=self.F_("chico"), text_color=C["suave"]),
                          self.resultado, 20, 640).pack(pady=(2, 0))

    def alternar_toma(self):
        if self.grabando:
            if self.detener_evt:
                self.detener_evt.set()
        else:
            self.grabar_toma()

    def grabar_toma(self):
        p = self.pasos[self.indice]
        self.grabando = True
        self.detener_evt = threading.Event()
        self.btn_toma.poner(modo="parar", activo=False)
        self._mostrar_resultado()
        self.btn_volver.configure(state="disabled")
        inicio = time.time()
        self._satura_hasta = 0.0
        minimo = p["min_voz"] + 3 if p.get("libre") else 2

        def reloj():
            if not self.grabando or not self.lbl_cuenta.winfo_exists():
                return
            t = time.time() - inicio
            if time.time() < self._satura_hasta:
                self.lbl_cuenta.configure(text="Muy fuerte: aléjate un poco del micrófono", text_color=C["rojo"])
            else:
                self.lbl_cuenta.configure(text=f"Grabando · {mmss(t)} de {mmss(p['segundos'])}", text_color=C["rojo"])
            self.btn_toma.poner(activo=t >= minimo)
            self.after(200, reloj)
        reloj()

        def trabajo():
            audio = self.app.motor.grabar_toma(p["segundos"], self.mic_id,
                                               lambda r: self.app.post(self.medidor.poner, r, True),
                                               self.detener_evt, self._marcar_saturado)
            self.app.post(self.lbl_cuenta.configure, text="Revisando la toma...", text_color=C["suave"])
            return audio, self.app.motor.analizar_toma(audio, p)
        self.app.en_hilo(trabajo, lambda r: self._toma_lista(p["id"], *r), self._toma_error)

    def _marcar_saturado(self):
        self._satura_hasta = time.time() + 2.0      # el aviso se queda 2 s para alcanzar a leerlo

    def _fin_grabacion(self):
        self.grabando = False
        self.btn_volver.configure(state="normal")
        if self.btn_toma.winfo_exists():
            self.btn_toma.poner(modo="grabar", activo=True)

    def _toma_error(self, msg):
        self._fin_grabacion()
        self._mostrar_resultado()
        self.app.error(f"No se pudo grabar:\n\n{msg}")

    def _toma_lista(self, pid, audio, info):
        self._fin_grabacion()
        self.tomas[pid], self.info[pid] = audio, info
        if not self.lbl_cuenta.winfo_exists():
            return
        self.medidor.reiniciar()
        voz = f"{info['voz_seg']:.0f} s de voz"
        self.lbl_cuenta.configure(text=f"Te escuché bien · {voz}" if info["calidad"] != "mala" else voz,
                                  text_color=C["suave"])
        self.ir_a(self.indice)

    def escuchar(self):
        pid = self.pasos[self.indice]["id"]
        if pid in self.tomas:
            try:
                self.app.motor.reproducir(self.tomas[pid])
            except Exception as e:
                self.app.error(f"No se pudo reproducir: {e}")

    def siguiente(self):
        pendientes = [i for i in range(self.indice + 1, len(self.pasos)) if self.pasos[i]["id"] not in self.info]
        faltan = [i for i, p in enumerate(self.pasos) if p["id"] not in self.info]
        if pendientes:
            self.ir_a(pendientes[0])
        elif faltan:
            self.ir_a(faltan[0])
        else:
            self.vista_revision()

    # ---------------------------------------------------------------- revisión
    def vista_revision(self):
        self.limpiar()
        c = self.centro(700)
        self.rev = tarjeta(c, corner_radius=20)
        self.rev.pack(fill="x", pady=(30, 0))
        ctk.CTkLabel(self.rev, text=f"Huella de voz de {self.nombre}", font=self.F_("grande"),
                     text_color=C["texto"]).pack(anchor="w", padx=34, pady=(30, 2))
        self.lbl_rev = ctk.CTkLabel(self.rev, text="Comparando las tomas entre sí...", font=self.F_("normal"),
                                    text_color=C["suave"])
        self.lbl_rev.pack(anchor="w", padx=34, pady=(0, 30))
        self.btn_volver.configure(state="disabled")
        self.app.en_hilo(lambda: self.app.motor.revisar_tomas(self.nombre, self.tomas), self._revision_lista,
                         self._revision_error)

    def _revision_error(self, msg):
        self.btn_volver.configure(state="normal")
        self.lbl_rev.configure(text="No se pudo revisar.", text_color=C["rojo"])
        self.app.error(msg)

    def _revision_lista(self, rep):
        self.reporte = rep
        self.btn_volver.configure(state="normal")
        malas_calidad = [p for p in self.pasos if self.info[p["id"]]["calidad"] == "mala"]
        inconsistentes = {pid for pid, v in rep["consistencia"].items() if v < mt.CONSISTENCIA_MINIMA}
        seg = sum(self.info[p["id"]]["voz_seg"] for p in self.pasos)
        self.lbl_rev.configure(text=f"{mmss(seg)} de voz en {len(self.pasos)} tomas")
        self.lbl_rev.pack_configure(pady=(0, 14))
        lista = ctk.CTkFrame(self.rev, fg_color="transparent", width=1, height=1)
        lista.pack(fill="x", padx=34)
        lista.grid_columnconfigure((0, 1, 2), weight=1, uniform="tomas")
        for k, p in enumerate(self.pasos):
            info = self.info[p["id"]]
            mala = info["calidad"] == "mala" or p["id"] in inconsistentes
            icono = self.app.I["error"] if mala else (self.app.I["aviso"] if info["calidad"] == "regular"
                                                      else self.app.I["check"])
            ctk.CTkButton(lista, text="  " + p["titulo"], image=icono, compound="left", anchor="w", height=38,
                          corner_radius=10, fg_color=C["rojo_suave"] if mala else C["fondo"],
                          hover_color=C["gris_hover"], text_color=C["texto"], font=self.F_("chico"),
                          command=lambda i=k: self.ir_a(i)).grid(row=k // 3, column=k % 3, sticky="ew", padx=4,
                                                                 pady=4)
        avisos = list(rep["avisos"])
        if malas_calidad:
            avisos.insert(0, "Repite las tomas marcadas en rojo (haz clic en ellas).")
        if rep.get("parecidos"):
            s, n = rep["parecidos"][0]
            if s < mt.PARECIDO_ALTO:
                avisos.append(f"La voz más parecida es la de {n} ({s:.2f}): se distinguen bien.")
        for a in avisos:
            ajustar_texto(ctk.CTkLabel(self.rev, text=a, font=self.F_("chico"), text_color=C["suave"],
                                       justify="left", anchor="w"), self.rev, 80, 600).pack(anchor="w", padx=34,
                                                                                         pady=(10, 0))
        hay_problemas = bool(malas_calidad) or not rep["ok"]
        if not avisos:
            ctk.CTkLabel(self.rev, text="  Todas las tomas son consistentes", image=self.app.I["check"],
                         compound="left", font=self.F_("negrita"), text_color=C["verde"]).pack(anchor="w", padx=34,
                                                                                            pady=(14, 0))
        self.btn_guardar = boton(self.rev, "Guardar huella de voz", lambda: self.guardar(hay_problemas),
                                 "secundario" if hay_problemas else "principal", font=self.F_("boton"), height=48)
        self.btn_guardar.pack(fill="x", padx=34, pady=(24, 30))

    def guardar(self, con_problemas):
        if con_problemas and not messagebox.askyesno(
                "Enrolar", "Hay tomas con problemas y el reconocimiento puede fallar. ¿Guardar de todos modos?",
                parent=self.app):
            return
        self.btn_guardar.configure(state="disabled", text="Guardando y recalculando...")
        self.btn_volver.configure(state="disabled")
        malas = {pid for pid, v in self.reporte["consistencia"].items() if v < mt.CONSISTENCIA_MINIMA}
        malas |= {p["id"] for p in self.pasos if self.info[p["id"]]["calidad"] == "mala"}
        buenas = {pid: a for pid, a in self.tomas.items() if pid not in malas} or self.tomas
        self.app.en_hilo(lambda: self.app.motor.guardar_enrolamiento(self.nombre, buenas), self._guardado,
                         self._guardado_error)

    def _guardado_error(self, msg):
        self.btn_volver.configure(state="normal")
        self.btn_guardar.configure(state="normal", text="Guardar huella de voz")
        self.app.error(msg)

    def _guardado(self, _):
        self.btn_volver.configure(state="normal")
        self.app.personas_cambiaron()
        self.tomas, self.info = {}, {}
        self.limpiar()
        self.lbl_quien.configure(text="")
        c = self.centro(560)
        t = tarjeta(c, corner_radius=20)
        t.pack(fill="x", pady=(60, 0))
        ctk.CTkLabel(t, text="", image=_icono("check", C["verde"], 40), width=72, height=72, corner_radius=36,
                     fg_color=C["verde_suave"]).pack(pady=(36, 10))
        ctk.CTkLabel(t, text=f"Se guardó la voz de {self.nombre}", font=self.F_("grande"),
                     text_color=C["texto"]).pack()
        ctk.CTkLabel(t, text="La app ya la reconoce en las conversaciones.", font=self.F_("normal"),
                     text_color=C["suave"]).pack(pady=(4, 0))
        fila = ctk.CTkFrame(t, fg_color="transparent", width=1, height=1)
        fila.pack(pady=(22, 34))
        boton(fila, "Enrolar a otra persona", lambda: self.vista_inicio(), "secundario",
              font=self.F_("boton_normal"), height=44).pack(side="left", padx=6)
        boton(fila, "Listo", lambda: self.app.mostrar("personas"), "principal", font=self.F_("boton"), height=44,
              width=110).pack(side="left", padx=6)


def usar_entorno_virtual():
    """Si se abrió con el Python del sistema, se vuelve a abrir con el de .venv
    (por si se olvida activar el entorno o se abre con doble clic)."""
    if CONGELADA or sys.prefix != sys.base_prefix:
        return
    for partes in ((".venv", "Scripts", "python.exe"), (".venv", "bin", "python")):
        py = os.path.join(AQUI, *partes)
        if os.path.exists(py) and os.path.abspath(py) != os.path.abspath(sys.executable):
            import subprocess
            sys.exit(subprocess.call([py, os.path.abspath(__file__)] + sys.argv[1:]))


def main():
    import multiprocessing
    multiprocessing.freeze_support()
    usar_entorno_virtual()
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)   # texto nítido en pantallas con escala
        except Exception:
            pass
    simulado = "--simulado" in sys.argv
    app = App(mt.MotorSimulado() if simulado else mt.Motor())
    app.mainloop()


if __name__ == "__main__":
    main()

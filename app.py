#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcriptor multihablante — aplicación de escritorio.

  python app.py              con el motor real (micrófono, SpeechBrain y Whisper)
  python app.py --simulado   sin modelos ni micrófono, para trabajar en la interfaz

Tres secciones:
  Conversación   grabar y ver la transcripción en vivo; al detener, pasada final más precisa
  Personas       quién está enrolado; probar el reconocimiento; borrar
  Enrolar voz    enrolamiento guiado con 9 tomas en distintos tonos y revisión de calidad
"""
import os
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

AQUI = os.path.dirname(os.path.abspath(__file__))
os.chdir(AQUI)                       # voces/, galeria.npz y modelos/ son relativos a esta carpeta
sys.path.insert(0, AQUI)

import numpy as np   # noqa: E402

import motor as mt   # noqa: E402

# ------------------------------------------------------------------ colores y tipografía
C = {
    "fondo": "#F4F5F7", "tarjeta": "#FFFFFF", "borde": "#E2E5EA", "texto": "#1C2330", "suave": "#6B7380",
    "lateral": "#161B26", "lateral_hover": "#232A38", "lateral_activo": "#2C3546", "lateral_texto": "#C9D0DC",
    "acento": "#2F6FEB", "acento_hover": "#2559C4", "rojo": "#E5484D", "rojo_hover": "#C93C41",
    "verde": "#2F9E62", "ambar": "#D98E04", "gris_btn": "#E9ECF1", "gris_btn_hover": "#DDE1E8",
    "medidor_fondo": "#E6E9EE",
}
COLORES_HABLANTES = ["#2F6FEB", "#D6457A", "#1F9D8B", "#C46A12", "#7A4FD1", "#3E8E2F", "#B23A3A", "#0E7BB8"]
COLOR_DESCONOCIDO = ["#6B7380", "#8A6F4E", "#5C6E91", "#7B6A87"]


def familia():
    disponibles = set(tkfont.families())
    for f in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Ubuntu", "DejaVu Sans"):
        if f in disponibles:
            return f
    return "TkDefaultFont"


class Boton(tk.Label):
    """Botón plano con hover y estado deshabilitado (los de ttk se ven anticuados en Windows)."""

    ESTILOS = {
        "primario": ("acento", "acento_hover", "#FFFFFF"),
        "rojo": ("rojo", "rojo_hover", "#FFFFFF"),
        "gris": ("gris_btn", "gris_btn_hover", "texto"),
    }

    def __init__(self, padre, texto, comando, estilo="gris", fuente=None, padx=14, pady=7):
        fondo, hover, tinta = self.ESTILOS[estilo]
        self._f, self._h = C[fondo], C[hover]
        self._t = C.get(tinta, tinta)
        super().__init__(padre, text=texto, bg=self._f, fg=self._t, font=fuente, padx=padx, pady=pady,
                         cursor="hand2")
        self.comando, self.activo = comando, True
        self.bind("<Enter>", lambda e: self.activo and self.config(bg=self._h))
        self.bind("<Leave>", lambda e: self.config(bg=self._f if self.activo else C["borde"]))
        self.bind("<Button-1>", lambda e: self.activo and self.comando and self.comando())

    def estilo(self, estilo):
        fondo, hover, tinta = self.ESTILOS[estilo]
        self._f, self._h, self._t = C[fondo], C[hover], C.get(tinta, tinta)
        self.habilitar(self.activo)

    def habilitar(self, si=True):
        self.activo = bool(si)
        self.config(bg=self._f if si else C["borde"], fg=self._t if si else "#9AA1AC",
                    cursor="hand2" if si else "arrow")


class Medidor(tk.Canvas):
    """Barra de nivel del micrófono."""

    def __init__(self, padre, ancho=170, alto=10, fondo=None):
        super().__init__(padre, width=ancho, height=alto, bg=fondo or C["tarjeta"], highlightthickness=0)
        self.ancho, self.alto = ancho, alto
        self.create_rectangle(0, 0, ancho, alto, fill=C["medidor_fondo"], width=0)
        self.barra = self.create_rectangle(0, 0, 0, alto, fill=C["verde"], width=0)
        self.valor = 0.0

    def reiniciar(self):
        self.valor = 0.0
        self.coords(self.barra, 0, 0, 0, self.alto)

    def poner(self, rms, hablando=False):
        db = 20 * np.log10(rms + 1e-9)
        objetivo = float(np.clip((db + 62) / 47, 0, 1))
        self.valor = objetivo if objetivo > self.valor else self.valor * 0.75 + objetivo * 0.25
        color = C["rojo"] if db > -3 else (C["acento"] if hablando else C["verde"])
        self.coords(self.barra, 0, 0, self.ancho * self.valor, self.alto)
        self.itemconfig(self.barra, fill=color)


def ajustar(etiqueta, margen=4):
    """El texto de la etiqueta se acomoda al ancho disponible (sin cortarse a la derecha)."""
    etiqueta.bind("<Configure>", lambda e: etiqueta.config(wraplength=max(e.width - margen, 120)))
    return etiqueta


def mmss(seg):
    seg = int(max(seg, 0))
    return f"{seg // 60:02d}:{seg % 60:02d}"


# ====================================================================== aplicación

class App(tk.Tk):
    def __init__(self, motor):
        super().__init__()
        self.motor = motor
        self.listo = False
        self.q = queue.Queue()
        self.title("Transcriptor multihablante")
        self.geometry("1180x740")
        self.minsize(960, 620)
        self.configure(bg=C["fondo"])
        fam = familia()
        self.F = {
            "titulo": (fam, 18, "bold"), "sub": (fam, 10), "normal": (fam, 10), "negrita": (fam, 10, "bold"),
            "chico": (fam, 9), "nav": (fam, 11), "marca": (fam, 13, "bold"), "lectura": (fam, 16),
            "boton": (fam, 10, "bold"), "grande": (fam, 12, "bold"), "transcripcion": (fam, 11),
            "nombre": (fam, 11, "bold"), "tiempo": (fam, 9),
        }
        self._estilos_ttk()
        self._armar()
        self.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.after(40, self._bombear)
        self.mostrar("conversacion")
        self.en_hilo(lambda: self.motor.cargar(lambda t: self.post(self.estado_modelos, t, False)),
                     self._modelos_listos, self._modelos_error)

    # ---------------------------------------------------------------- utilidades de hilos
    def post(self, funcion, *args):
        """Ejecuta funcion(*args) en el hilo de la interfaz."""
        self.q.put((funcion, args))

    def _bombear(self):
        try:
            while True:
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

    def _estilos_ttk(self):
        st = ttk.Style(self)
        st.theme_use("clam")
        st.configure("Treeview", rowheight=30, font=self.F["normal"], background=C["tarjeta"],
                     fieldbackground=C["tarjeta"], bordercolor=C["borde"], borderwidth=0)
        st.configure("Treeview.Heading", font=self.F["negrita"], background=C["fondo"], foreground=C["suave"],
                     relief="flat", borderwidth=0)
        st.map("Treeview", background=[("selected", "#DCE7FD")], foreground=[("selected", C["texto"])])
        st.configure("TCombobox", padding=5)
        st.configure("Horizontal.TProgressbar", troughcolor=C["medidor_fondo"], background=C["acento"],
                     bordercolor=C["medidor_fondo"], lightcolor=C["acento"], darkcolor=C["acento"], thickness=8)
        st.configure("TCheckbutton", background=C["fondo"], foreground=C["texto"], font=self.F["normal"])
        st.map("TCheckbutton", background=[("active", C["fondo"])])
        st.configure("Tarjeta.TCheckbutton", background=C["tarjeta"])
        st.map("Tarjeta.TCheckbutton", background=[("active", C["tarjeta"])])
        st.configure("TRadiobutton", background=C["tarjeta"], font=self.F["chico"])
        st.map("TRadiobutton", background=[("active", C["tarjeta"])])
        st.configure("Fondo.TRadiobutton", background=C["fondo"], font=self.F["normal"])
        st.map("Fondo.TRadiobutton", background=[("active", C["fondo"])])
        st.configure("Vertical.TScrollbar", troughcolor=C["tarjeta"], background=C["gris_btn"],
                     bordercolor=C["tarjeta"], arrowcolor=C["suave"])

    # ---------------------------------------------------------------- estructura
    def _armar(self):
        lateral = tk.Frame(self, bg=C["lateral"], width=210)
        lateral.pack(side="left", fill="y")
        lateral.pack_propagate(False)
        tk.Label(lateral, text="Transcriptor", font=self.F["marca"], fg="#FFFFFF", bg=C["lateral"],
                 anchor="w").pack(fill="x", padx=20, pady=(22, 0))
        tk.Label(lateral, text=f"multihablante · v{mt.tr.VERSION}", font=self.F["chico"], fg="#7D8696",
                 bg=C["lateral"], anchor="w").pack(fill="x", padx=20, pady=(0, 22))
        self.nav = {}
        for clave, texto in (("conversacion", "Conversación"), ("personas", "Personas"), ("enrolar", "Enrolar voz")):
            b = tk.Label(lateral, text="   " + texto, font=self.F["nav"], fg=C["lateral_texto"], bg=C["lateral"],
                         anchor="w", padx=12, pady=10, cursor="hand2")
            b.pack(fill="x", padx=10, pady=1)
            b.bind("<Button-1>", lambda e, k=clave: self.mostrar(k))
            b.bind("<Enter>", lambda e, w=b, k=clave: self.actual != k and w.config(bg=C["lateral_hover"]))
            b.bind("<Leave>", lambda e, w=b, k=clave: self.actual != k and w.config(bg=C["lateral"]))
            self.nav[clave] = b
        pie = tk.Frame(lateral, bg=C["lateral"])
        pie.pack(side="bottom", fill="x", padx=20, pady=18)
        if self.motor.simulado:
            tk.Label(pie, text=" MODO SIMULADO ", font=self.F["chico"], fg="#161B26", bg="#F2C14E").pack(anchor="w",
                                                                                                         pady=(0, 10))
        fila = tk.Frame(pie, bg=C["lateral"])
        fila.pack(fill="x")
        self.punto = tk.Canvas(fila, width=10, height=10, bg=C["lateral"], highlightthickness=0)
        self.punto.pack(side="left", pady=4)
        self.punto_id = self.punto.create_oval(1, 1, 9, 9, fill=C["ambar"], width=0)
        tk.Label(fila, text="Modelos", font=self.F["negrita"], fg="#FFFFFF", bg=C["lateral"]).pack(side="left",
                                                                                                    padx=6)
        self.lbl_modelos = tk.Label(pie, text="Cargando...", font=self.F["chico"], fg="#9AA3B2", bg=C["lateral"],
                                    anchor="w", justify="left", wraplength=170)
        self.lbl_modelos.pack(fill="x", pady=(4, 0))

        self.contenido = tk.Frame(self, bg=C["fondo"])
        self.contenido.pack(side="left", fill="both", expand=True)
        self.paginas = {
            "conversacion": PaginaConversacion(self.contenido, self),
            "personas": PaginaPersonas(self.contenido, self),
            "enrolar": PaginaEnrolar(self.contenido, self),
        }
        for p in self.paginas.values():
            p.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.actual = None

    def mostrar(self, clave):
        if self.actual == "enrolar" and clave != "enrolar" and self.paginas["enrolar"].grabando:
            return
        if clave == "enrolar" and self.motor.en_curso():
            messagebox.showinfo("Transcriptor", "Detén la conversación antes de enrolar a alguien.", parent=self)
            return
        self.actual = clave
        for k, b in self.nav.items():
            b.config(bg=C["lateral_activo"] if k == clave else C["lateral"],
                     fg="#FFFFFF" if k == clave else C["lateral_texto"])
        self.paginas[clave].tkraise()
        self.paginas[clave].al_mostrar()

    def estado_modelos(self, texto, listo):
        self.lbl_modelos.config(text=texto)
        self.punto.itemconfig(self.punto_id, fill=C["verde"] if listo else C["ambar"])

    def _modelos_listos(self, descripcion):
        self.listo = True
        self.estado_modelos(f"Listos · {descripcion}", True)
        for p in self.paginas.values():
            p.al_cambiar_modelos()

    def _modelos_error(self, msg):
        self.estado_modelos("No se pudieron cargar", False)
        self.punto.itemconfig(self.punto_id, fill=C["rojo"])
        self.error("No se pudieron cargar los modelos:\n\n" + msg +
                   "\n\nRevisa que el entorno virtual esté activado (.venv) y que la instalación esté completa.")

    def personas_cambiaron(self):
        for p in self.paginas.values():
            p.al_cambiar_personas()

    def cerrar(self):
        conv = self.paginas["conversacion"]
        if self.motor.en_curso() or conv.ocupado:
            if not messagebox.askyesno("Transcriptor", "Hay una grabación o transcripción en curso. ¿Salir de todos modos?",
                                       parent=self):
                return
            try:
                self.motor.detener()
            except Exception:
                pass
        self.destroy()

    # ---------------------------------------------------------------- piezas comunes
    def encabezado(self, padre, titulo, subtitulo):
        f = tk.Frame(padre, bg=C["fondo"])
        f.pack(fill="x", padx=32, pady=(26, 14))
        tk.Label(f, text=titulo, font=self.F["titulo"], fg=C["texto"], bg=C["fondo"], anchor="w").pack(fill="x")
        tk.Label(f, text=subtitulo, font=self.F["sub"], fg=C["suave"], bg=C["fondo"], anchor="w").pack(fill="x")
        return f

    def tarjeta(self, padre, **kw):
        return tk.Frame(padre, bg=C["tarjeta"], highlightbackground=C["borde"], highlightcolor=C["borde"],
                        highlightthickness=1, **kw)

    def combo_microfono(self, padre, fondo):
        f = tk.Frame(padre, bg=fondo)
        tk.Label(f, text="Micrófono", font=self.F["chico"], fg=C["suave"], bg=fondo).pack(side="left", padx=(0, 6))
        combo = ttk.Combobox(f, state="readonly", width=34, font=self.F["chico"])
        combo.pack(side="left")
        f.combo = combo
        f.ids = []

        def rellenar():
            try:
                lista, pred = self.motor.microfonos()
            except Exception as e:
                lista, pred = [], None
                combo.set(f"(sin micrófonos: {e})")
            f.ids = [i for i, _ in lista]
            combo["values"] = [n for _, n in lista]
            if lista:
                combo.current(f.ids.index(pred) if pred in f.ids else 0)

        def elegido():
            i = combo.current()
            return f.ids[i] if 0 <= i < len(f.ids) else None
        f.rellenar, f.elegido = rellenar, elegido
        return f


class Pagina(tk.Frame):
    def __init__(self, padre, app):
        super().__init__(padre, bg=C["fondo"])
        self.app = app

    def al_mostrar(self):
        pass

    def al_cambiar_modelos(self):
        pass

    def al_cambiar_personas(self):
        pass


# ====================================================================== conversación

class PaginaConversacion(Pagina):
    def __init__(self, padre, app):
        super().__init__(padre, app)
        self.lineas = {"vivo": [], "final": []}
        self.vista = tk.StringVar(value="vivo")
        self.colores = {}
        self.ocupado = False
        self.inicio = None
        self.ruta_audio = None
        app.encabezado(self, "Conversación", "Graba una plática y mira quién dijo qué, en tiempo real.")

        barra = app.tarjeta(self)
        barra.pack(fill="x", padx=32)
        izq = tk.Frame(barra, bg=C["tarjeta"])
        izq.pack(side="left", padx=16, pady=14)
        self.btn_grabar = Boton(izq, "●  Grabar", self.alternar, "rojo", app.F["boton"], padx=18, pady=9)
        self.btn_grabar.pack(side="left")
        self.lbl_reloj = tk.Label(izq, text="00:00", font=app.F["grande"], fg=C["texto"], bg=C["tarjeta"])
        self.lbl_reloj.pack(side="left", padx=(16, 10))
        self.medidor = Medidor(izq, 150, 10)
        self.medidor.pack(side="left")
        self.lbl_estado = tk.Label(izq, text="Esperando modelos...", font=app.F["chico"], fg=C["suave"],
                                   bg=C["tarjeta"], width=22, anchor="w")
        self.lbl_estado.pack(side="left", padx=12)
        self.mic = app.combo_microfono(barra, C["tarjeta"])
        self.mic.combo.config(width=28)
        self.mic.pack(side="right", padx=16)

        opciones = tk.Frame(self, bg=C["fondo"])
        opciones.pack(fill="x", padx=32, pady=(10, 8))
        self.var_final = tk.BooleanVar(value=True)
        self.var_ingles = tk.BooleanVar(value=True)
        ttk.Checkbutton(opciones, text="Pasada final al detener (más precisa)", variable=self.var_final).pack(
            side="left")
        ttk.Checkbutton(opciones, text="Términos técnicos en inglés", variable=self.var_ingles).pack(side="left",
                                                                                                   padx=18)
        self.selector = tk.Frame(opciones, bg=C["fondo"])
        tk.Label(self.selector, text="Ver:", font=app.F["chico"], fg=C["suave"], bg=C["fondo"]).pack(side="left")
        for valor, texto in (("vivo", "En vivo"), ("final", "Final")):
            ttk.Radiobutton(self.selector, text=texto, value=valor, variable=self.vista, style="Fondo.TRadiobutton",
                            command=self.redibujar).pack(side="left", padx=4)

        cuerpo = tk.Frame(self, bg=C["fondo"])
        cuerpo.pack(fill="both", expand=True, padx=32, pady=(0, 24))
        lado = app.tarjeta(cuerpo, width=220)
        lado.pack(side="right", fill="y", padx=(14, 0))
        lado.pack_propagate(False)
        tk.Label(lado, text="Participantes", font=app.F["negrita"], fg=C["texto"], bg=C["tarjeta"],
                 anchor="w").pack(fill="x", padx=16, pady=(14, 6))
        self.lista_part = tk.Frame(lado, bg=C["tarjeta"])
        self.lista_part.pack(fill="x", padx=16)
        self.lbl_enrolados = tk.Label(lado, text="", font=app.F["chico"], fg=C["suave"], bg=C["tarjeta"],
                                      justify="left", anchor="w", wraplength=186)
        self.lbl_enrolados.pack(side="bottom", fill="x", padx=16, pady=14)

        hoja = app.tarjeta(cuerpo)
        hoja.pack(side="left", fill="both", expand=True)
        cab = tk.Frame(hoja, bg=C["tarjeta"])
        cab.pack(fill="x", padx=18, pady=(12, 0))
        self.lbl_vista = tk.Label(cab, text="Transcripción en vivo", font=app.F["negrita"], fg=C["texto"],
                                  bg=C["tarjeta"])
        self.lbl_vista.pack(side="left")
        acciones = tk.Frame(cab, bg=C["tarjeta"])
        acciones.pack(side="right")
        chico = app.F["chico"]
        self.btn_abrir = Boton(acciones, "Abrir audio…", self.abrir_audio, "gris", chico, padx=10, pady=4)
        self.btn_guardar = Boton(acciones, "Guardar TXT", self.guardar, "gris", chico, padx=10, pady=4)
        self.btn_copiar = Boton(acciones, "Copiar", self.copiar, "gris", chico, padx=10, pady=4)
        self.btn_limpiar = Boton(acciones, "Limpiar", self.limpiar, "gris", chico, padx=10, pady=4)
        for btn in (self.btn_abrir, self.btn_guardar, self.btn_copiar, self.btn_limpiar):
            btn.pack(side="left", padx=(6, 0))
        self.progreso = tk.Frame(hoja, bg=C["tarjeta"])
        self.lbl_progreso = tk.Label(self.progreso, text="", font=app.F["chico"], fg=C["acento"], bg=C["tarjeta"],
                                     anchor="w")
        self.lbl_progreso.pack(fill="x")
        self.barra_prog = ttk.Progressbar(self.progreso, mode="determinate", maximum=1.0)
        self.barra_prog.pack(fill="x", pady=(4, 0))
        marco = tk.Frame(hoja, bg=C["tarjeta"])
        marco.pack(fill="both", expand=True, padx=(18, 4), pady=(6, 12))
        self.texto = tk.Text(marco, wrap="word", font=app.F["transcripcion"], bg=C["tarjeta"], fg=C["texto"],
                             relief="flat", highlightthickness=0, padx=4, pady=6, spacing1=1, spacing3=2,
                             cursor="arrow")
        scroll = ttk.Scrollbar(marco, orient="vertical", command=self.texto.yview)
        self.texto.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.texto.pack(side="left", fill="both", expand=True)
        self.texto.tag_configure("tiempo", font=app.F["tiempo"], foreground="#9AA1AC")
        self.texto.tag_configure("cuerpo", lmargin1=0, lmargin2=0, spacing3=12)
        self.texto.tag_configure("solape", font=app.F["chico"], foreground=C["ambar"])
        self.texto.tag_configure("vacio", font=app.F["sub"], foreground="#A0A7B2", justify="center",
                                 spacing1=120)
        self.redibujar()
        self._actualizar_botones()     # todo deshabilitado hasta que carguen los modelos
        self._tic()

    # ---------------------------------------------------------------- estado
    def al_cambiar_modelos(self):
        self.mic.rellenar()
        self._actualizar_botones()
        self.al_cambiar_personas()

    def al_cambiar_personas(self):
        if not self.app.listo:
            return
        try:
            nombres = [p["nombre"] for p in self.app.motor.personas()]
        except Exception:
            nombres = []
        if nombres:
            self.lbl_enrolados.config(text="Enrolados: " + ", ".join(nombres) +
                                      "\nLas demás voces aparecen como «Desconocido N».")
        else:
            self.lbl_enrolados.config(text="Nadie está enrolado todavía. Ve a «Enrolar voz» para que el "
                                           "sistema reconozca a cada persona por su nombre.")

    def _actualizar_botones(self):
        listo, grabando = self.app.listo, self.app.motor.en_curso()
        self.btn_grabar.habilitar(listo and not self.ocupado)
        for b in (self.btn_abrir, self.btn_limpiar):
            b.habilitar(listo and not grabando and not self.ocupado)
        hay = bool(self.lineas["vivo"] or self.lineas["final"])
        self.btn_guardar.habilitar(hay and not grabando)
        self.btn_copiar.habilitar(hay)
        if listo and not grabando and not self.ocupado:
            self.lbl_estado.config(text="Listo para grabar", fg=C["suave"])

    def _tic(self):
        if self.inicio is not None:
            self.lbl_reloj.config(text=mmss(time.time() - self.inicio))
        self.after(250, self._tic)

    # ---------------------------------------------------------------- grabar / detener
    def alternar(self):
        if self.app.motor.en_curso():
            self.detener()
        else:
            self.grabar()

    def grabar(self):
        if (self.lineas["vivo"] or self.lineas["final"]) and not messagebox.askyesno(
                "Nueva conversación", "Se borrará la transcripción que está en pantalla. ¿Continuar?",
                parent=self.app):
            return
        self.lineas = {"vivo": [], "final": []}
        self.vista.set("vivo")
        self.colores = {}
        self.redibujar()
        try:
            self.ruta_audio = self.app.motor.iniciar(self.mic.elegido(), self._evento_motor,
                                                     espanglish=self.var_ingles.get())
        except Exception as e:
            self.app.error(f"No se pudo empezar a grabar:\n\n{e}")
            return
        self.inicio = time.time()
        self.btn_grabar.config(text="■  Detener")
        self.btn_grabar.estilo("primario")
        self.lbl_estado.config(text="Escuchando...", fg=C["rojo"])
        self._actualizar_botones()

    def _evento_motor(self, tipo, datos):
        # llega desde los hilos del motor: se pasa al hilo de la interfaz
        if tipo == "nivel":
            self.app.post(self.medidor.poner, datos["rms"], datos["hablando"])
        elif tipo == "linea":
            self.app.post(self.agregar_linea, datos, "vivo")
        elif tipo == "aviso":
            self.app.post(self._aviso, datos)

    def _aviso(self, mensaje):
        self.lbl_estado.config(text="Revisa el aviso", fg=C["ambar"])
        messagebox.showwarning("Transcriptor", mensaje, parent=self.app)

    def detener(self):
        self.ocupado = True
        self.btn_grabar.habilitar(False)
        self.lbl_estado.config(text="Terminando lo pendiente...", fg=C["suave"])
        self.app.en_hilo(self.app.motor.detener, self._detenido, self._fallo_detener)

    def _fallo_detener(self, msg):
        self._restaurar_boton()
        self.app.error(msg)

    def _restaurar_boton(self):
        self.inicio = None
        self.ocupado = False
        self.btn_grabar.config(text="●  Grabar")
        self.btn_grabar.estilo("rojo")
        self.medidor.reiniciar()
        self._actualizar_botones()

    def _detenido(self, ruta):
        self._restaurar_boton()
        self.ruta_audio = ruta
        if self.var_final.get() and ruta:
            self.correr_pasada_final(ruta)

    # ---------------------------------------------------------------- pasada final
    def correr_pasada_final(self, ruta, titulo="Pasada final"):
        self.ocupado = True
        self._actualizar_botones()
        self.lbl_estado.config(text=f"{titulo}...", fg=C["acento"])
        self.progreso.pack(fill="x", padx=18, pady=(8, 0), before=self.texto.master)
        self.barra_prog["value"] = 0
        self.lbl_progreso.config(text=f"{titulo}: preparando...")

        def avance(etapa, fraccion):
            total = 0.3 * fraccion if etapa.startswith("Separando") else 0.3 + 0.7 * fraccion
            self.app.post(self._avance, f"{titulo}: {etapa.lower()}... {int(total * 100)} %", total)

        espanglish = self.var_ingles.get()
        self.app.en_hilo(lambda: self.app.motor.pasada_final(ruta, avance, espanglish),
                         self._final_listo, self._final_error)

    def _avance(self, texto, valor):
        self.lbl_progreso.config(text=texto)
        self.barra_prog["value"] = valor

    def _final_listo(self, lineas):
        self.progreso.pack_forget()
        self.ocupado = False
        self.lineas["final"] = lineas
        if lineas:
            self.vista.set("final")
        self.redibujar()
        self.texto.see("1.0")
        self._actualizar_botones()
        self.lbl_estado.config(text="Transcripción final lista" if lineas else "No se detectó voz", fg=C["verde"])

    def _final_error(self, msg):
        self.progreso.pack_forget()
        self.ocupado = False
        self._actualizar_botones()
        self.app.error(f"La pasada final falló:\n\n{msg}\n\nLa transcripción en vivo sigue disponible.")

    def abrir_audio(self):
        ruta = filedialog.askopenfilename(
            parent=self.app, title="Elige un audio para transcribir",
            initialdir=os.path.join(AQUI, "transcripciones") if os.path.isdir("transcripciones") else AQUI,
            filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.opus *.webm *.aac"), ("Todos", "*.*")])
        if not ruta:
            return
        self.lineas = {"vivo": [], "final": []}
        self.colores = {}
        self.ruta_audio = ruta
        self.vista.set("final")
        self.redibujar()
        self.correr_pasada_final(ruta, "Transcribiendo " + os.path.basename(ruta))

    # ---------------------------------------------------------------- dibujo
    def color(self, nombre):
        if nombre not in self.colores:
            if nombre.startswith("Desconocido"):
                n = sum(1 for k in self.colores if k.startswith("Desconocido"))
                self.colores[nombre] = COLOR_DESCONOCIDO[n % len(COLOR_DESCONOCIDO)]
            else:
                n = sum(1 for k in self.colores if not k.startswith("Desconocido"))
                self.colores[nombre] = COLORES_HABLANTES[n % len(COLORES_HABLANTES)]
            self.texto.tag_configure(self._tag(nombre), foreground=self.colores[nombre], font=self.app.F["nombre"])
        return self.colores[nombre]

    @staticmethod
    def _tag(nombre):
        return "n_" + "".join(c if c.isalnum() else "_" for c in nombre)

    def agregar_linea(self, linea, cual):
        lista = self.lineas[cual]
        anterior = lista[-1] if lista else None
        # misma persona y pausa corta: se junta en el mismo párrafo
        if anterior and anterior["nombre"] == linea["nombre"] and linea["inicio"] - anterior["fin"] < 1.5:
            anterior["texto"] += " " + linea["texto"]
            anterior["fin"] = linea["fin"]
            anterior["solape"] = anterior["solape"] or linea["solape"]
        else:
            lista.append(dict(linea))
        if self.vista.get() == cual:
            self.redibujar()
        self._actualizar_botones()

    def redibujar(self):
        cual = self.vista.get()
        lineas = self.lineas[cual]
        abajo = self.texto.yview()[1] > 0.98
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        if not lineas:
            if self.ocupado or self.app.motor.en_curso():
                msg = "Escuchando... la transcripción aparecerá aquí en cuanto alguien hable."
            else:
                msg = "Presiona «Grabar» para empezar, o abre un audio ya grabado."
            self.texto.insert("end", msg, "vacio")
        for l in lineas:
            self.color(l["nombre"])
            self.texto.insert("end", l["nombre"], self._tag(l["nombre"]))
            self.texto.insert("end", f"   {mmss(l['inicio'])} – {mmss(l['fin'])}\n", "tiempo")
            self.texto.insert("end", l["texto"], "cuerpo")
            if l.get("solape"):
                self.texto.insert("end", "  ⚠ posible solapamiento con otra voz", "solape")
            self.texto.insert("end", "\n", "cuerpo")
        self.texto.configure(state="disabled")
        if abajo or self.app.motor.en_curso():
            self.texto.see("end")
        hay_final = bool(self.lineas["final"])
        if hay_final and self.lineas["vivo"]:
            self.selector.pack(side="right")
        else:
            self.selector.pack_forget()
        self.lbl_vista.config(text="Transcripción final (alta precisión)" if cual == "final" else
                              "Transcripción en vivo")
        self._participantes(lineas)

    def _participantes(self, lineas):
        for w in self.lista_part.winfo_children():
            w.destroy()
        tiempos = {}
        for l in lineas:
            tiempos[l["nombre"]] = tiempos.get(l["nombre"], 0.0) + max(l["fin"] - l["inicio"], 0)
        total = sum(tiempos.values()) or 1.0
        if not tiempos:
            tk.Label(self.lista_part, text="Aún nadie ha hablado.", font=self.app.F["chico"], fg=C["suave"],
                     bg=C["tarjeta"], anchor="w").pack(fill="x")
        for nombre, t in sorted(tiempos.items(), key=lambda kv: -kv[1]):
            fila = tk.Frame(self.lista_part, bg=C["tarjeta"])
            fila.pack(fill="x", pady=4)
            cv = tk.Canvas(fila, width=10, height=10, bg=C["tarjeta"], highlightthickness=0)
            cv.create_oval(1, 1, 9, 9, fill=self.color(nombre), width=0)
            cv.grid(row=0, column=0, padx=(0, 7))
            tk.Label(fila, text=nombre, font=self.app.F["negrita"], fg=C["texto"], bg=C["tarjeta"],
                     anchor="w").grid(row=0, column=1, sticky="w")
            tk.Label(fila, text=f"{mmss(t)} · {int(100 * t / total)} % del tiempo", font=self.app.F["chico"],
                     fg=C["suave"], bg=C["tarjeta"], anchor="w").grid(row=1, column=1, sticky="w")

    # ---------------------------------------------------------------- exportar
    def _texto_plano(self):
        return "\n".join(mt.formatear(l) for l in self.lineas[self.vista.get()])

    def guardar(self):
        if not self.lineas[self.vista.get()]:
            return
        base = os.path.splitext(os.path.basename(self.ruta_audio or "conversacion"))[0]
        sufijo = "_final" if self.vista.get() == "final" else ""
        ruta = filedialog.asksaveasfilename(
            parent=self.app, title="Guardar transcripción", defaultextension=".txt",
            initialdir=os.path.join(AQUI, "transcripciones") if os.path.isdir("transcripciones") else AQUI,
            initialfile=f"{base}{sufijo}.txt", filetypes=[("Texto", "*.txt")])
        if ruta:
            with open(ruta, "w", encoding="utf-8") as f:
                f.write(self._texto_plano() + "\n")
            self.lbl_estado.config(text="Guardado", fg=C["verde"])

    def copiar(self):
        self.app.clipboard_clear()
        self.app.clipboard_append(self._texto_plano())
        self.lbl_estado.config(text="Copiado al portapapeles", fg=C["verde"])

    def limpiar(self):
        self.lineas = {"vivo": [], "final": []}
        self.colores = {}
        self.vista.set("vivo")
        self.lbl_reloj.config(text="00:00")
        self.redibujar()
        self._actualizar_botones()


# ====================================================================== personas

class PaginaPersonas(Pagina):
    def __init__(self, padre, app):
        super().__init__(padre, app)
        app.encabezado(self, "Personas", "Voces enroladas: el sistema reconoce a estas personas por su nombre.")
        acciones = tk.Frame(self, bg=C["fondo"])
        acciones.pack(fill="x", padx=32, pady=(0, 12))
        self.btn_nueva = Boton(acciones, "+  Enrolar persona", lambda: self.app.paginas["enrolar"].nueva(),
                               "primario", app.F["boton"])
        self.btn_nueva.pack(side="left")
        self.btn_reenrolar = Boton(acciones, "Volver a enrolar", self.reenrolar, "gris", app.F["normal"])
        self.btn_reenrolar.pack(side="left", padx=8)
        self.btn_borrar = Boton(acciones, "Borrar", self.borrar, "gris", app.F["normal"])
        self.btn_borrar.pack(side="left")

        cuerpo = tk.Frame(self, bg=C["fondo"])
        cuerpo.pack(fill="both", expand=True, padx=32, pady=(0, 24))
        tabla = app.tarjeta(cuerpo)
        tabla.pack(side="left", fill="both", expand=True)
        self.tree = ttk.Treeview(tabla, columns=("nombre", "tomas", "audio", "estado"), show="headings",
                                 selectmode="browse")
        for col, txt, ancho in (("nombre", "Nombre", 220), ("tomas", "Tomas", 90), ("audio", "Audio grabado", 130),
                                ("estado", "Estado", 160)):
            self.tree.heading(col, text=txt, anchor="w")
            self.tree.column(col, width=ancho, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=1, pady=1)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._botones())
        self.lbl_info = ajustar(tk.Label(tabla, text="", font=app.F["chico"], fg=C["suave"], bg=C["tarjeta"],
                                         justify="left", anchor="w"))
        self.lbl_info.pack(fill="x", padx=16, pady=12)

        self.panel = app.tarjeta(cuerpo, width=320)
        self.panel.pack(side="right", fill="y", padx=(14, 0))
        self.panel.pack_propagate(False)
        tk.Label(self.panel, text="Probar reconocimiento", font=app.F["negrita"], fg=C["texto"], bg=C["tarjeta"],
                 anchor="w").pack(fill="x", padx=16, pady=(14, 2))
        ajustar(tk.Label(self.panel, text="Presiona el botón, habla unos 5 segundos y el sistema dirá a quién "
                                          "se parece tu voz.", font=app.F["chico"], fg=C["suave"], bg=C["tarjeta"],
                         justify="left", anchor="w")).pack(fill="x", padx=16)
        self.btn_probar = Boton(self.panel, "●  Hablar 5 segundos", self.probar, "primario", app.F["boton"])
        self.btn_probar.pack(anchor="w", padx=16, pady=(12, 0))
        self.medidor = Medidor(self.panel, 280, 8)
        self.medidor.pack(padx=16, pady=(12, 4), anchor="w")
        self.lbl_prueba = tk.Label(self.panel, text="", font=app.F["grande"], fg=C["texto"], bg=C["tarjeta"],
                                   anchor="w")
        self.lbl_prueba.pack(fill="x", padx=16, pady=(8, 4))
        self.barras = tk.Frame(self.panel, bg=C["tarjeta"])
        self.barras.pack(fill="x", padx=16)
        self.al_cambiar_personas()

    def al_mostrar(self):
        self.al_cambiar_personas()

    def al_cambiar_modelos(self):
        self.al_cambiar_personas()

    def al_cambiar_personas(self):
        self.tree.delete(*self.tree.get_children())
        if not self.app.listo:
            self.lbl_info.config(text="Cargando modelos...")
            self._botones()
            return
        personas = self.app.motor.personas()
        for p in personas:
            estado = "Lista" if p["en_galeria"] else "Falta recalcular"
            self.tree.insert("", "end", iid=p["nombre"],
                             values=(p["nombre"], p["tomas"], f"{p['segundos']:.0f} s", estado))
        g = getattr(self.app.motor, "galeria", None)
        umbrales = self.app.motor.umbrales() if personas else {}
        if not personas:
            info = "Todavía no hay nadie enrolado. Empieza con «Enrolar persona»."
        else:
            info = ("Umbral de cada persona: " + ", ".join(f"{n} {u:.2f}" for n, u in umbrales.items()) +
                    ". Se calcula solo, comparando las voces de todos: una voz muy distinta a las demás "
                    "necesita menos parecido para reconocerse que dos voces parecidas entre sí.")
        if g is not None and g is not True and personas:
            S = g.centroides @ g.centroides.T
            parecidas = [f"{g.nombres[i]} y {g.nombres[j]} ({S[i, j]:.2f})" for i in range(len(g.nombres))
                         for j in range(i + 1, len(g.nombres)) if S[i, j] > mt.PARECIDO_ALTO]
            if parecidas:
                info += "\n⚠ Se parecen mucho y podrían confundirse: " + ", ".join(parecidas) + "."
        self.lbl_info.config(text=info)
        self._botones()

    def _sel(self):
        s = self.tree.selection()
        return s[0] if s else None

    def _botones(self):
        listo = self.app.listo
        self.btn_nueva.habilitar(listo)
        self.btn_reenrolar.habilitar(listo and self._sel() is not None)
        self.btn_borrar.habilitar(listo and self._sel() is not None)
        self.btn_probar.habilitar(listo and bool(self.tree.get_children()))

    def reenrolar(self):
        if self._sel():
            self.app.paginas["enrolar"].nueva(self._sel())

    def borrar(self):
        nombre = self._sel()
        if not nombre or not messagebox.askyesno("Borrar persona", f"¿Borrar la voz de {nombre}? "
                                                 "Tendrá que volver a enrolarse.", parent=self.app):
            return
        self.lbl_info.config(text=f"Borrando a {nombre} y recalculando...")
        self.app.en_hilo(lambda: self.app.motor.borrar_persona(nombre), lambda r: self.app.personas_cambiaron())

    def probar(self):
        self.btn_probar.habilitar(False)
        self.lbl_prueba.config(text="Habla ahora...", fg=C["rojo"])
        for w in self.barras.winfo_children():
            w.destroy()
        mic = self.app.paginas["conversacion"].mic.elegido()

        def trabajo():
            audio = self.app.motor.grabar_toma(5, mic, lambda r: self.app.post(self.medidor.poner, r, True))
            self.app.post(self.lbl_prueba.config, {"text": "Analizando...", "fg": C["suave"]})
            return self.app.motor.identificar(audio)
        self.app.en_hilo(trabajo, self._resultado, self._fallo)

    def _fallo(self, msg):
        self.lbl_prueba.config(text="")
        self.medidor.reiniciar()
        self._botones()
        self.app.error(msg)

    def _resultado(self, r):
        sims, niveles = r
        self.medidor.reiniciar()
        self._botones()
        mejor, nombre = sims[0]
        umbral = niveles.get(nombre, 0.4)
        segundo = sims[1][0] if len(sims) > 1 else -1
        if mejor >= umbral and mejor - segundo >= 0.05:
            self.lbl_prueba.config(text=f"Eres {nombre}", fg=C["verde"])
        elif mejor >= umbral:
            self.lbl_prueba.config(text=f"¿{nombre}? (dudoso)", fg=C["ambar"])
        else:
            self.lbl_prueba.config(text="Voz desconocida", fg=C["suave"])
        for s, n in sims[:5]:
            umbral = niveles.get(n, 0.4)
            fila = tk.Frame(self.barras, bg=C["tarjeta"])
            fila.pack(fill="x", pady=3)
            tk.Label(fila, text=n, font=self.app.F["chico"], fg=C["texto"], bg=C["tarjeta"], width=12,
                     anchor="w").pack(side="left")
            cv = tk.Canvas(fila, width=150, height=10, bg=C["tarjeta"], highlightthickness=0)
            cv.pack(side="left", padx=4)
            cv.create_rectangle(0, 0, 150, 10, fill=C["medidor_fondo"], width=0)
            cv.create_rectangle(0, 0, 150 * float(np.clip(s, 0, 1)), 10,
                                fill=C["verde"] if s >= umbral else "#B9C0CC", width=0)
            x = 150 * umbral
            cv.create_line(x, 0, x, 10, fill=C["texto"], width=1)
            tk.Label(fila, text=f"{s:.2f}", font=self.app.F["chico"], fg=C["suave"], bg=C["tarjeta"]).pack(
                side="left")
        tk.Label(self.barras, text="La línea marca el umbral de cada persona.", font=self.app.F["chico"],
                 fg=C["suave"], bg=C["tarjeta"], anchor="w").pack(fill="x", pady=(6, 0))


# ====================================================================== enrolamiento guiado

ICONOS = {"pendiente": ("○", "#A0A7B2"), "actual": ("●", C["acento"]), "buena": ("✓", C["verde"]),
          "regular": ("!", C["ambar"]), "mala": ("✗", C["rojo"])}


class PaginaEnrolar(Pagina):
    def __init__(self, padre, app):
        super().__init__(padre, app)
        self.pasos = app.motor.pasos
        self.grabando = False
        self.tomas, self.info = {}, {}
        self.nombre, self.indice = None, 0
        self.detener_evt = None
        app.encabezado(self, "Enrolar voz", "Enseña al sistema cómo suena una persona, en 9 tomas cortas "
                                            "con distintos tonos (unos 4 minutos).")
        self.area = tk.Frame(self, bg=C["fondo"])
        self.area.pack(fill="both", expand=True, padx=32, pady=(0, 24))
        self.vista_inicio()

    def limpiar(self):
        for w in self.area.winfo_children():
            w.destroy()

    def al_cambiar_modelos(self):
        if hasattr(self, "btn_comenzar") and self.btn_comenzar.winfo_exists():
            self.btn_comenzar.habilitar(True)

    # ---------------------------------------------------------------- inicio
    def nueva(self, nombre=None):
        if self.grabando:
            return
        self.app.mostrar("enrolar")
        self.vista_inicio(nombre)

    def vista_inicio(self, nombre=None):
        self.limpiar()
        self.tomas, self.info = {}, {}
        t = self.app.tarjeta(self.area)
        t.pack(fill="both", expand=True)
        cont = tk.Frame(t, bg=C["tarjeta"])
        cont.pack(fill="both", expand=True, padx=36, pady=30)
        tk.Label(cont, text="¿Quién se va a enrolar?", font=self.app.F["grande"], fg=C["texto"],
                 bg=C["tarjeta"], anchor="w").pack(fill="x")
        fila = tk.Frame(cont, bg=C["tarjeta"])
        fila.pack(fill="x", pady=(10, 4))
        self.var_nombre = tk.StringVar(value=nombre or "")
        e = tk.Entry(fila, textvariable=self.var_nombre, font=self.app.F["lectura"], width=24, relief="solid",
                     bd=1, highlightthickness=0)
        e.pack(side="left", ipady=4)
        e.focus_set()
        e.bind("<Return>", lambda ev: self.comenzar())
        self.btn_comenzar = Boton(fila, "Comenzar  →", self.comenzar, "primario", self.app.F["boton"], padx=18,
                                  pady=9)
        self.btn_comenzar.pack(side="left", padx=12)
        self.btn_comenzar.habilitar(self.app.listo)
        if nombre:
            tk.Label(cont, text=f"Se reemplazarán las tomas anteriores de {nombre}.", font=self.app.F["chico"],
                     fg=C["ambar"], bg=C["tarjeta"], anchor="w").pack(fill="x")
        self.mic = self.app.combo_microfono(cont, C["tarjeta"])
        self.mic.pack(anchor="w", pady=(12, 0))
        if self.app.listo:
            self.mic.rellenar()
        tk.Frame(cont, bg=C["borde"], height=1).pack(fill="x", pady=22)
        tk.Label(cont, text="Cómo funciona", font=self.app.F["negrita"], fg=C["texto"], bg=C["tarjeta"],
                 anchor="w").pack(fill="x")
        consejos = [
            "Vas a leer 8 textos cortos con distintos tonos (normal, pregunta, emoción, voz baja, voz fuerte, "
            "rápido, lento y con términos técnicos) y luego platicar libremente unos 15 segundos.",
            "Cada toma se revisa al momento: volumen, saturación, ruido de fondo y segundos de voz. "
            "Si sale mal, la repites.",
            "Al final se comparan las tomas entre sí (para detectar si habló otra persona) y con las "
            "voces ya enroladas (para avisar si dos personas suenan casi igual).",
            "Busca un lugar callado y quédate a la distancia del micrófono a la que vas a hablar en la "
            "conversación. Solo debe hablar la persona que se enrola.",
        ]
        for c in consejos:
            f = tk.Frame(cont, bg=C["tarjeta"])
            f.pack(fill="x", pady=4)
            tk.Label(f, text="•", font=self.app.F["normal"], fg=C["acento"], bg=C["tarjeta"]).pack(side="left",
                                                                                                 anchor="n")
            ajustar(tk.Label(f, text=c, font=self.app.F["normal"], fg=C["texto"], bg=C["tarjeta"], justify="left",
                             anchor="w")).pack(side="left", fill="x", expand=True, padx=8)

    def comenzar(self):
        if not self.app.listo:
            return
        nombre, err = mt.nombre_valido(self.var_nombre.get())
        if err:
            messagebox.showwarning("Enrolar voz", err, parent=self.app)
            return
        existentes = [p["nombre"].lower() for p in self.app.motor.personas()]
        if nombre.lower() in existentes and not messagebox.askyesno(
                "Enrolar voz", f"{nombre} ya está enrolado. ¿Reemplazar su voz con este nuevo enrolamiento?",
                parent=self.app):
            return
        self.nombre, self.mic_id = nombre, self.mic.elegido()
        self.tomas, self.info = {}, {}
        self.vista_pasos()
        self.ir_a(0)

    # ---------------------------------------------------------------- pasos
    def vista_pasos(self):
        self.limpiar()
        izq = self.app.tarjeta(self.area, width=250)
        izq.pack(side="left", fill="y")
        izq.pack_propagate(False)
        tk.Label(izq, text=self.nombre, font=self.app.F["grande"], fg=C["texto"], bg=C["tarjeta"],
                 anchor="w").pack(fill="x", padx=18, pady=(16, 0))
        tk.Label(izq, text="Tomas", font=self.app.F["chico"], fg=C["suave"], bg=C["tarjeta"], anchor="w").pack(
            fill="x", padx=18, pady=(0, 8))
        self.filas_pasos = []
        for i, p in enumerate(self.pasos):
            f = tk.Frame(izq, bg=C["tarjeta"], cursor="hand2")
            f.pack(fill="x", padx=10)
            ico = tk.Label(f, text="○", font=self.app.F["negrita"], fg="#A0A7B2", bg=C["tarjeta"], width=2)
            ico.pack(side="left")
            lbl = tk.Label(f, text=f"{i + 1}. {p['titulo']}", font=self.app.F["normal"], fg=C["texto"],
                           bg=C["tarjeta"], anchor="w", pady=6)
            lbl.pack(side="left", fill="x", expand=True)
            for w in (f, ico, lbl):
                w.bind("<Button-1>", lambda e, k=i: not self.grabando and self.ir_a(k))
            self.filas_pasos.append((f, ico, lbl))
        abajo = tk.Frame(izq, bg=C["tarjeta"])
        abajo.pack(side="bottom", fill="x", padx=18, pady=16)
        self.btn_revisar = Boton(abajo, "Revisar y guardar", self.vista_revision, "primario", self.app.F["boton"])
        self.btn_revisar.pack(fill="x")
        self.btn_cancelar = Boton(abajo, "Cancelar", self.cancelar, "gris", self.app.F["normal"])
        self.btn_cancelar.pack(fill="x", pady=(8, 0))

        self.der = self.app.tarjeta(self.area)
        self.der.pack(side="left", fill="both", expand=True, padx=(14, 0))
        self._iconos()

    def _estado_paso(self, i):
        pid = self.pasos[i]["id"]
        if pid in self.info:
            return self.info[pid]["calidad"]
        return "actual" if i == self.indice else "pendiente"

    def _iconos(self):
        for i, (f, ico, lbl) in enumerate(self.filas_pasos):
            est = self._estado_paso(i)
            simbolo, color = ICONOS[est]
            fondo = "#EEF3FE" if i == self.indice else C["tarjeta"]
            for w in (f, ico, lbl):
                w.config(bg=fondo)
            ico.config(text=simbolo, fg=color)
            lbl.config(font=self.app.F["negrita"] if i == self.indice else self.app.F["normal"])
        completas = sum(1 for p in self.pasos if p["id"] in self.info)
        self.btn_revisar.habilitar(completas == len(self.pasos) and not self.grabando)
        self.btn_cancelar.habilitar(not self.grabando)

    def ir_a(self, i):
        self.indice = i
        p = self.pasos[i]
        for w in self.der.winfo_children():
            w.destroy()
        cont = tk.Frame(self.der, bg=C["tarjeta"])
        cont.pack(fill="both", expand=True, padx=34, pady=26)
        tk.Label(cont, text=f"PASO {i + 1} DE {len(self.pasos)}", font=self.app.F["chico"], fg=C["acento"],
                 bg=C["tarjeta"], anchor="w").pack(fill="x")
        tk.Label(cont, text=p["titulo"], font=self.app.F["titulo"], fg=C["texto"], bg=C["tarjeta"],
                 anchor="w").pack(fill="x")
        ajustar(tk.Label(cont, text=p["instruccion"], font=self.app.F["normal"], fg=C["suave"], bg=C["tarjeta"],
                         anchor="w", justify="left")).pack(fill="x", pady=(2, 14))
        lectura = tk.Frame(cont, bg="#F7F9FC", highlightbackground=C["borde"], highlightthickness=1)
        lectura.pack(fill="x")
        estilo_texto = self.app.F["normal"] if p.get("libre") else self.app.F["lectura"]
        self.lbl_lectura = tk.Label(lectura, text=p["texto"], font=estilo_texto, fg=C["texto"], bg="#F7F9FC",
                                    justify="left", anchor="w", wraplength=640)
        self.lbl_lectura.pack(fill="x", padx=22, pady=20)
        self.lbl_lectura.bind("<Configure>", lambda e: self.lbl_lectura.config(wraplength=max(e.width - 10, 200)))

        controles = tk.Frame(cont, bg=C["tarjeta"])
        controles.pack(fill="x", pady=(18, 0))
        self.btn_toma = Boton(controles, "●  Grabar", self.grabar_toma, "rojo", self.app.F["boton"], padx=18, pady=9)
        self.btn_toma.pack(side="left")
        self.btn_terminar = Boton(controles, "Terminar", self.terminar_toma, "gris", self.app.F["normal"])
        self.btn_terminar.habilitar(False)
        self.btn_terminar.pack(side="left", padx=8)
        self.medidor = Medidor(controles, 180, 10)
        self.medidor.pack(side="left", padx=12)
        self.lbl_cuenta = tk.Label(controles, text=f"hasta {p['segundos']} s", font=self.app.F["chico"],
                                   fg=C["suave"], bg=C["tarjeta"])
        self.lbl_cuenta.pack(side="left")
        self.barra = ttk.Progressbar(cont, mode="determinate", maximum=p["segundos"])
        self.barra.pack(fill="x", pady=(12, 0))

        self.resultado = tk.Frame(cont, bg=C["tarjeta"])
        self.resultado.pack(fill="x", pady=(16, 0))
        nav = tk.Frame(cont, bg=C["tarjeta"])
        nav.pack(side="bottom", fill="x")
        self.btn_ant = Boton(nav, "←  Anterior", lambda: self.ir_a(self.indice - 1), "gris", self.app.F["normal"])
        self.btn_ant.pack(side="left")
        self.btn_ant.habilitar(i > 0)
        self.btn_sig = Boton(nav, "Siguiente  →", self.siguiente, "primario", self.app.F["boton"])
        self.btn_sig.pack(side="right")
        self.btn_escuchar = Boton(nav, "▶  Escuchar", self.escuchar, "gris", self.app.F["normal"])
        self.btn_escuchar.pack(side="right", padx=8)
        self._mostrar_resultado()
        self._iconos()

    def _mostrar_resultado(self):
        for w in self.resultado.winfo_children():
            w.destroy()
        p = self.pasos[self.indice]
        info = self.info.get(p["id"])
        self.btn_sig.habilitar(bool(info) and not self.grabando)
        self.btn_escuchar.habilitar(p["id"] in self.tomas and not self.grabando)
        if p["id"] in self.tomas and not self.grabando:
            self.btn_toma.config(text="↻  Repetir")
        if self.indice == len(self.pasos) - 1:
            self.btn_sig.config(text="Revisar  →")
        if not info:
            return
        simbolo, color = ICONOS[info["calidad"]]
        titulo = {"buena": "Toma buena", "regular": "Toma aceptable", "mala": "Hay que repetir esta toma"}
        tk.Label(self.resultado, text=f"{simbolo}  {titulo[info['calidad']]}", font=self.app.F["negrita"],
                 fg=color, bg=C["tarjeta"], anchor="w").pack(fill="x")
        datos = f"{info['voz_seg']:.0f} s de voz · volumen {20 * np.log10(info['pico'] + 1e-9):.0f} dB"
        if info.get("snr_db") is not None:
            datos += f" · voz {info['snr_db']:.0f} dB arriba del ruido"
        tk.Label(self.resultado, text=datos, font=self.app.F["chico"], fg=C["suave"], bg=C["tarjeta"],
                 anchor="w").pack(fill="x")
        for a in info["avisos"]:
            ajustar(tk.Label(self.resultado, text="• " + a, font=self.app.F["normal"], fg=C["texto"],
                             bg=C["tarjeta"], anchor="w", justify="left")).pack(fill="x", pady=(4, 0))

    def grabar_toma(self):
        p = self.pasos[self.indice]
        self.grabando = True
        self.detener_evt = threading.Event()
        self.btn_toma.habilitar(False)
        self.btn_terminar.habilitar(False)
        for b in (self.btn_ant, self.btn_sig, self.btn_escuchar):
            b.habilitar(False)
        self._iconos()
        inicio = time.time()
        minimo = p["min_voz"] + 3 if p.get("libre") else 2

        def reloj():
            if not self.grabando or not self.lbl_cuenta.winfo_exists():
                return
            t = time.time() - inicio
            self.barra["value"] = min(t, p["segundos"])
            self.lbl_cuenta.config(text=f"{t:4.0f} / {p['segundos']} s", fg=C["rojo"])
            self.btn_terminar.habilitar(t >= minimo)
            self.after(200, reloj)
        reloj()

        def trabajo():
            audio = self.app.motor.grabar_toma(p["segundos"], self.mic_id,
                                               lambda r: self.app.post(self.medidor.poner, r, True),
                                               self.detener_evt)
            self.app.post(self.lbl_cuenta.config, {"text": "Revisando la toma...", "fg": C["suave"]})
            return audio, self.app.motor.analizar_toma(audio, p)
        self.app.en_hilo(trabajo, lambda r: self._toma_lista(p["id"], *r), self._toma_error)

    def terminar_toma(self):
        if self.detener_evt:
            self.detener_evt.set()

    def _toma_error(self, msg):
        self.grabando = False
        self.btn_toma.habilitar(True)
        self.btn_terminar.habilitar(False)
        self.btn_ant.habilitar(self.indice > 0)
        self._mostrar_resultado()
        self._iconos()
        self.app.error(f"No se pudo grabar:\n\n{msg}")

    def _toma_lista(self, pid, audio, info):
        self.grabando = False
        self.tomas[pid], self.info[pid] = audio, info
        if not self.lbl_cuenta.winfo_exists():
            return
        self.medidor.reiniciar()
        self.lbl_cuenta.config(text=f"{info['duracion']:.0f} s grabados", fg=C["suave"])
        self.btn_toma.habilitar(True)
        self.btn_terminar.habilitar(False)
        self.btn_ant.habilitar(self.indice > 0)
        self._mostrar_resultado()
        self._iconos()

    def escuchar(self):
        pid = self.pasos[self.indice]["id"]
        if pid in self.tomas:
            try:
                self.app.motor.reproducir(self.tomas[pid])
            except Exception as e:
                self.app.error(f"No se pudo reproducir: {e}")

    def siguiente(self):
        if self.indice + 1 < len(self.pasos):
            # salta al siguiente paso que falte
            pendientes = [i for i in range(self.indice + 1, len(self.pasos)) if self.pasos[i]["id"] not in self.info]
            self.ir_a(pendientes[0] if pendientes else self.indice + 1)
        else:
            faltan = [i for i, p in enumerate(self.pasos) if p["id"] not in self.info]
            if faltan:
                self.ir_a(faltan[0])
            else:
                self.vista_revision()

    def cancelar(self):
        if self.tomas and not messagebox.askyesno("Enrolar voz", "¿Descartar las tomas grabadas?", parent=self.app):
            return
        self.vista_inicio()

    # ---------------------------------------------------------------- revisión
    def vista_revision(self):
        for w in self.der.winfo_children():
            w.destroy()
        self.indice = -1
        self._iconos()
        self.btn_revisar.habilitar(False)
        cont = tk.Frame(self.der, bg=C["tarjeta"])
        cont.pack(fill="both", expand=True, padx=34, pady=26)
        tk.Label(cont, text="REVISIÓN", font=self.app.F["chico"], fg=C["acento"], bg=C["tarjeta"],
                 anchor="w").pack(fill="x")
        tk.Label(cont, text=f"Huella de voz de {self.nombre}", font=self.app.F["titulo"], fg=C["texto"],
                 bg=C["tarjeta"], anchor="w").pack(fill="x")
        self.lbl_rev = tk.Label(cont, text="Comparando las tomas entre sí...", font=self.app.F["normal"],
                                fg=C["suave"], bg=C["tarjeta"], anchor="w")
        self.lbl_rev.pack(fill="x", pady=(2, 12))
        self.rev_cuerpo = tk.Frame(cont, bg=C["tarjeta"])
        self.rev_cuerpo.pack(fill="both", expand=True)
        self.rev_nav = tk.Frame(cont, bg=C["tarjeta"])
        self.rev_nav.pack(side="bottom", fill="x")
        self.btn_cancelar.habilitar(False)
        self.app.en_hilo(lambda: self.app.motor.revisar_tomas(self.nombre, self.tomas), self._revision_lista,
                         self._revision_error)

    def _revision_error(self, msg):
        self.btn_cancelar.habilitar(True)
        self.lbl_rev.config(text="No se pudo revisar.", fg=C["rojo"])
        self.app.error(msg)

    def _revision_lista(self, rep):
        self.reporte = rep
        self.btn_cancelar.habilitar(True)
        malas_calidad = [p for p in self.pasos if self.info[p["id"]]["calidad"] == "mala"]
        seg = sum(self.info[p["id"]]["voz_seg"] for p in self.pasos)
        self.lbl_rev.config(text=f"{seg:.0f} s de voz en {len(self.pasos)} tomas.", fg=C["suave"])
        tabla = tk.Frame(self.rev_cuerpo, bg=C["tarjeta"])
        tabla.pack(fill="x")
        for col, txt in enumerate(("Toma", "Calidad", "Parecido con las demás tomas")):
            tk.Label(tabla, text=txt, font=self.app.F["chico"], fg=C["suave"], bg=C["tarjeta"], anchor="w").grid(
                row=0, column=col, sticky="w", padx=(0, 24), pady=(0, 4))
        for r, p in enumerate(self.pasos, start=1):
            info = self.info[p["id"]]
            simbolo, color = ICONOS[info["calidad"]]
            enlace = tk.Label(tabla, text=p["titulo"], font=self.app.F["normal"], fg=C["acento"], bg=C["tarjeta"],
                              anchor="w", cursor="hand2")
            enlace.grid(row=r, column=0, sticky="w", padx=(0, 24), pady=2)
            enlace.bind("<Button-1>", lambda e, k=r - 1: self._volver_a_paso(k))
            tk.Label(tabla, text=f"{simbolo} {info['calidad']}", font=self.app.F["normal"], fg=color,
                     bg=C["tarjeta"], anchor="w").grid(row=r, column=1, sticky="w", padx=(0, 24))
            v = rep["consistencia"].get(p["id"])
            celda = tk.Frame(tabla, bg=C["tarjeta"])
            celda.grid(row=r, column=2, sticky="w")
            if v is not None:
                cv = tk.Canvas(celda, width=160, height=9, bg=C["tarjeta"], highlightthickness=0)
                cv.pack(side="left")
                cv.create_rectangle(0, 0, 160, 9, fill=C["medidor_fondo"], width=0)
                ok = v >= mt.CONSISTENCIA_MINIMA
                cv.create_rectangle(0, 0, 160 * float(np.clip(v, 0, 1)), 9, fill=C["verde"] if ok else C["rojo"],
                                    width=0)
                tk.Label(celda, text=f" {v:.2f}", font=self.app.F["chico"], fg=C["suave"], bg=C["tarjeta"]).pack(
                    side="left")
            else:
                tk.Label(celda, text="sin voz suficiente", font=self.app.F["chico"], fg=C["rojo"],
                         bg=C["tarjeta"]).pack(side="left")
        avisos = list(rep["avisos"])
        if malas_calidad:
            avisos.insert(0, "Repite las tomas marcadas con ✗: " + ", ".join(p["titulo"] for p in malas_calidad) + ".")
        if rep.get("parecidos"):
            s, n = rep["parecidos"][0]
            nota = f"La voz más parecida entre las ya enroladas es {n} ({s:.2f})."
            nota += " Bien: se distinguen con claridad." if s < mt.PARECIDO_ALTO else ""
            ajustar(tk.Label(self.rev_cuerpo, text=nota, font=self.app.F["chico"], fg=C["suave"], bg=C["tarjeta"],
                             anchor="w", justify="left")).pack(fill="x", pady=(14, 0))
        for a in avisos:
            ajustar(tk.Label(self.rev_cuerpo, text="⚠  " + a, font=self.app.F["normal"], fg=C["ambar"],
                             bg=C["tarjeta"], anchor="w", justify="left")).pack(fill="x", pady=(8, 0))
        if not avisos:
            tk.Label(self.rev_cuerpo, text="✓  Todas las tomas son consistentes. Lista para guardar.",
                     font=self.app.F["negrita"], fg=C["verde"], bg=C["tarjeta"], anchor="w").pack(fill="x",
                                                                                                pady=(14, 0))
        hay_problemas = bool(malas_calidad) or not rep["ok"]
        Boton(self.rev_nav, "Guardar huella de voz", lambda: self.guardar(hay_problemas),
              "gris" if hay_problemas else "primario", self.app.F["boton"], padx=18, pady=9).pack(side="right")

    def _volver_a_paso(self, k):
        self.ir_a(k)

    def guardar(self, con_problemas):
        if con_problemas and not messagebox.askyesno(
                "Enrolar voz", "Hay tomas con problemas y el reconocimiento puede fallar. ¿Guardar de todos modos?",
                parent=self.app):
            return
        for w in self.rev_nav.winfo_children():
            w.destroy()
        self.btn_cancelar.habilitar(False)
        self.lbl_rev.config(text="Guardando y recalculando la galería de todas las personas...", fg=C["acento"])
        malas = {pid for pid, v in self.reporte["consistencia"].items() if v < mt.CONSISTENCIA_MINIMA}
        malas |= {p["id"] for p in self.pasos if self.info[p["id"]]["calidad"] == "mala"}
        buenas = {pid: a for pid, a in self.tomas.items() if pid not in malas} or self.tomas
        self.app.en_hilo(lambda: self.app.motor.guardar_enrolamiento(self.nombre, buenas), self._guardado,
                         self._guardado_error)

    def _guardado_error(self, msg):
        self.btn_cancelar.habilitar(True)
        self.lbl_rev.config(text="No se pudo guardar.", fg=C["rojo"])
        self.app.error(msg)

    def _guardado(self, _):
        self.app.personas_cambiaron()
        for w in self.area.winfo_children():
            w.destroy()
        t = self.app.tarjeta(self.area)
        t.pack(fill="both", expand=True)
        cont = tk.Frame(t, bg=C["tarjeta"])
        cont.place(relx=0.5, rely=0.42, anchor="center")
        tk.Label(cont, text="✓", font=(familia(), 40, "bold"), fg=C["verde"], bg=C["tarjeta"]).pack()
        tk.Label(cont, text=f"Se guardó la voz de {self.nombre}", font=self.app.F["titulo"], fg=C["texto"],
                 bg=C["tarjeta"]).pack(pady=(4, 2))
        tk.Label(cont, text="El sistema ya puede reconocer su voz en las conversaciones.", font=self.app.F["sub"],
                 fg=C["suave"], bg=C["tarjeta"]).pack()
        fila = tk.Frame(cont, bg=C["tarjeta"])
        fila.pack(pady=20)
        Boton(fila, "Probar reconocimiento", lambda: (self.app.mostrar("personas"),
                                                      self.app.paginas["personas"].probar()),
              "primario", self.app.F["boton"]).pack(side="left", padx=4)
        Boton(fila, "Enrolar a otra persona", lambda: self.vista_inicio(), "gris", self.app.F["normal"]).pack(
            side="left", padx=4)


def usar_entorno_virtual():
    """Si se abrió con el Python del sistema y existe .venv, se vuelve a abrir con el Python de .venv
    (así funciona aunque se olvide activar el entorno, o con doble clic en Transcriptor.bat)."""
    if sys.prefix != sys.base_prefix:
        return
    for partes in ((".venv", "Scripts", "python.exe"), (".venv", "bin", "python")):
        py = os.path.join(AQUI, *partes)
        if os.path.exists(py) and os.path.abspath(py) != os.path.abspath(sys.executable):
            import subprocess
            sys.exit(subprocess.call([py, os.path.abspath(__file__)] + sys.argv[1:]))


def main():
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

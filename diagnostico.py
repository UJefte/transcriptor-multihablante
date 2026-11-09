# -*- coding: utf-8 -*-
"""
Datos del equipo y paquete para soporte.

Sirve para que, cuando la app se prueba en otra computadora, sea fácil saber en qué se corrió
(procesador, memoria, tarjeta gráfica, versiones) y mandar los archivos de una prueba en un solo .zip.

El paquete NO lleva las voces enroladas. La última grabación solo se incluye si la persona lo elige.
"""
import datetime
import glob
import os
import platform
import subprocess
import sys
import zipfile


def _correr(comando, segundos=8):
    """Corre un comando sin abrir ventana y regresa lo que imprime (o "" si falla)."""
    try:
        extra = {"creationflags": 0x08000000} if os.name == "nt" else {}   # CREATE_NO_WINDOW
        r = subprocess.run(comando, capture_output=True, text=True, timeout=segundos, **extra)
        return r.stdout.strip()
    except Exception:
        return ""


def procesador():
    if os.name == "nt":
        try:
            import winreg
            clave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(clave, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linea in f:
                if linea.startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "Desconocido"


def memoria_gb():
    if os.name == "nt":
        try:
            import ctypes

            class Estado(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            e = Estado()
            e.dwLength = ctypes.sizeof(Estado)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(e))
            return e.ullTotalPhys / 1024 ** 3, e.ullAvailPhys / 1024 ** 3
        except Exception:
            pass
    try:
        datos = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for linea in f:
                k, v = linea.split(":", 1)
                datos[k] = int(v.split()[0]) / 1024 ** 2
        return datos["MemTotal"], datos.get("MemAvailable", 0.0)
    except Exception:
        return None, None


def graficas():
    """Nombres de las tarjetas gráficas (Intel, AMD y NVIDIA)."""
    if os.name == "nt":
        salida = _correr(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"])
        nombres = [l.strip() for l in salida.splitlines() if l.strip()]
        if nombres:
            return nombres
    salida = _correr(["lspci"])
    return [l.split(": ", 1)[1] for l in salida.splitlines() if "VGA" in l or "3D controller" in l] or ["Desconocida"]


def nvidia():
    """Nombre y memoria de la tarjeta NVIDIA si PyTorch la ve; None si no hay."""
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name} ({p.total_memory / 1024 ** 3:.1f} GB, CUDA {torch.version.cuda})"
    except Exception:
        pass
    return None


def openvino():
    """Dispositivos que ve OpenVINO (si está instalado). Para saber si se puede usar la gráfica Intel."""
    try:
        import openvino as ov
        nucleo = ov.Core()
        return ", ".join(f"{d}: {nucleo.get_property(d, 'FULL_DEVICE_NAME')}" for d in nucleo.available_devices)
    except Exception:
        return None


def versiones():
    from importlib import metadata
    salida = {"Python": platform.python_version()}
    for paquete in ("torch", "speechbrain", "faster-whisper", "ctranslate2", "silero-vad", "customtkinter",
                    "onnxruntime", "openvino", "openvino-genai", "transformers"):
        try:
            salida[paquete] = metadata.version(paquete)
        except Exception:
            pass
    return salida


def resumen(motor=None, version_app=""):
    """Diccionario corto para mostrar en la pantalla de Ajustes."""
    total, libre = memoria_gb()
    datos = {
        "Procesador": f"{procesador()} · {os.cpu_count()} hilos",
        "Memoria": f"{total:.0f} GB ({libre:.0f} GB libres)" if total else "Desconocida",
        "Gráfica": ", ".join(graficas()),
        "Sistema": f"{platform.system()} {platform.release()} ({platform.version()})",
    }
    nv = nvidia()
    datos["Tarjeta NVIDIA"] = nv or "No hay (o no la ve PyTorch)"
    if motor is not None:
        datos["Transcribiendo con"] = motor.equipo()
    return datos


def texto_completo(motor=None, version_app=""):
    """Todo en texto plano, para copiar y pegar."""
    lineas = [f"Transcriptor {version_app} · diagnóstico del {datetime.datetime.now():%Y-%m-%d %H:%M}", ""]
    for k, v in resumen(motor, version_app).items():
        lineas.append(f"{k}: {v}")
    ov = openvino()
    if ov:
        lineas.append(f"OpenVINO ve: {ov}")
    lineas.append("")
    lineas.append("Versiones: " + ", ".join(f"{k} {v}" for k, v in versiones().items()))
    if motor is not None:
        try:
            p = motor.perfil()
            lineas.append(f"Entorno elegido: {p['nombre']} (vad {p['vad']}, pausa {p['silencio_ms']} ms, "
                          f"penalización {p['penal']}, ajuste {p['ajuste']})")
            lineas.append("Ajustes: " + ", ".join(f"{k}={v}" for k, v in motor.ajustes.items()))
            umbrales = motor.umbrales()
            if umbrales:
                lineas.append("Personas enroladas y umbral: " + ", ".join(f"{n} {u:.2f}" for n, u in umbrales.items()))
            lista, pred = motor.microfonos()
            lineas.append("Micrófonos: " + "; ".join(n for _, n in lista))
        except Exception as e:
            lineas.append(f"(no se pudieron leer los datos del motor: {e})")
    return "\n".join(lineas)


def ultima_sesion(carpeta="transcripciones"):
    """Archivos de la grabación más reciente: (wav, txt en vivo, txt final), los que existan."""
    wavs = sorted(glob.glob(os.path.join(carpeta, "sesion_*.wav")), key=os.path.getmtime)
    if not wavs:
        return []
    base = os.path.splitext(wavs[-1])[0]
    return [r for r in (wavs[-1], base + ".txt", base + "_final.txt") if os.path.exists(r)]


def crear_paquete(destino, motor=None, version_app="", incluir_audio=False):
    """Arma un .zip con el diagnóstico, los ajustes, el registro y la última prueba."""
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("diagnostico.txt", texto_completo(motor, version_app))
        for archivo in ("ajustes.json", "vocabulario.txt", "vocabulario_tecnico.txt", "correcciones.txt"):
            if os.path.exists(archivo):
                z.write(archivo, archivo)
        if os.path.exists("transcriptor.log"):
            with open("transcriptor.log", encoding="utf-8", errors="replace") as f:
                z.writestr("transcriptor.log", "".join(f.readlines()[-400:]))   # solo lo más reciente
        for ruta in ultima_sesion():
            if ruta.endswith(".wav") and not incluir_audio:
                continue
            z.write(ruta, os.path.join("ultima_prueba", os.path.basename(ruta)))
    return destino


if __name__ == "__main__":
    print(texto_completo(version_app="(sin app)"))
    sys.exit(0)

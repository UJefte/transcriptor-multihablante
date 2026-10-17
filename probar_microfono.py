#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba rápida del micrófono.

  python probar_microfono.py            prueba el micrófono predeterminado de dos formas
  python probar_microfono.py --todos    prueba TODOS los micrófonos (habla sin parar mientras corre)
  python probar_microfono.py 3          prueba solo el dispositivo número 3
"""
import sys
import numpy as np
import sounddevice as sd


def nivel(x):
    x = np.asarray(x, dtype=np.float32).ravel()
    if not len(x):
        return "sin datos", 0.0
    db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
    pico = float(np.abs(x).max())
    estado = "OK, se escucha" if pico > 0.02 else ("muy bajo" if pico > 0.002 else "SILENCIO (no llega sonido)")
    return f"{db:5.0f} dB, pico {pico:.3f} -> {estado}", pico


def grabar_rec(dev, sr, seg):
    x = sd.rec(int(seg * sr), samplerate=sr, channels=1, dtype="float32", device=dev)
    sd.wait()
    return x


def grabar_stream(dev, sr, seg):
    buf = []
    with sd.InputStream(device=dev, samplerate=sr, channels=1, dtype="float32", blocksize=int(sr * 0.05),
                        callback=lambda i, f, t, s: buf.append(i[:, 0].copy())):
        sd.sleep(int(seg * 1000))
    return np.concatenate(buf) if buf else np.zeros(0)


def main():
    apis = sd.query_hostapis()
    if "--todos" in sys.argv:
        input("Presiona Enter y CUENTA EN VOZ ALTA sin parar (1, 2, 3...) hasta que termine la prueba...")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] < 1:
                continue
            sr = int(d["default_samplerate"])
            try:
                texto, _ = nivel(grabar_rec(i, sr, 1.5))
            except Exception as e:
                texto = f"no se pudo abrir ({str(e)[:50]})"
            print(f"  [{i:2d}] {d['name'][:40]:<40} {apis[d['hostapi']]['name']:<22} {texto}")
        return

    dev = int(sys.argv[1]) if len(sys.argv) > 1 else None
    nombre = sd.query_devices(dev if dev is not None else sd.default.device[0])["name"]
    print(f"Probando: {nombre}\n")
    for metodo, fn in (("grabación simple (como al enrolar)", grabar_rec),
                       ("flujo continuo (como en 'vivo')", grabar_stream)):
        input(f"Presiona Enter y habla 3 segundos -> {metodo}...")
        texto, _ = nivel(fn(dev, 16000, 3))
        print(f"  {texto}\n")


if __name__ == "__main__":
    main()

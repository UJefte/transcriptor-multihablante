#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara configuraciones de Whisper sobre UNA grabación del texto en espanglish y mide cada una.
No separa hablantes (solo texto), así cada prueba tarda poco.

Uso:
  python evaluacion\\experimento_espanglish.py transcripciones\\sesion_XXXX.wav
  python evaluacion\\experimento_espanglish.py transcripciones\\sesion_XXXX.wav --modelo small
"""
import argparse
import csv
import datetime
import os
import sys
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))   # transcriptor.py está en la carpeta de arriba
import transcriptor as tr
import evaluar as ev


# Nivela el volumen de cada tramo con voz
def nivelar_por_tramos(x):
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    reg = get_speech_timestamps(torch.from_numpy(x), load_silero_vad(), sampling_rate=tr.SR,
                                min_silence_duration_ms=1000, speech_pad_ms=100)
    y = x.copy()
    for r in reg:
        y[r["start"]:r["end"]] = tr.nivelar_volumen(x[r["start"]:r["end"]])
    return y


# Corre varias configuraciones de Whisper sobre el mismo audio y las compara
def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--modelo", default="large-v3-turbo")
    p.add_argument("--referencia", default=os.path.join(AQUI, "referencias", "espanglish.txt"))
    a = p.parse_args()

    asr, dev = tr.cargar_whisper(a.modelo, tr.dispositivo_torch())
    x = nivelar_por_tramos(tr.cargar_audio(a.audio))
    ref = [w for w, _ in ev.leer(a.referencia)]
    listas = [os.path.join("pruebas", "terminos_en_vocabulario.txt"), os.path.join("pruebas", "terminos_nuevos.txt")]
    tr.ESPANGLISH["activo"] = True
    vocab = tr.leer_vocabulario([])

    configs = [
        # nombre, prompt, hotwords, arrastrar texto previo, modo
        # (B y E se descartaron por bucles; F y G, por cortes a media oración)
        ("A. sin espanglish (como antes)", tr.PROMPT_SIMPLE, None, True, "completo"),
        ("C. espanglish sin arrastre", tr.PROMPT_ESPANGLISH, vocab, False, "completo"),
        ("H. espanglish por bloques (cortes en pausas)", tr.PROMPT_ESPANGLISH, vocab, False, "bloques"),
    ]
    with open(a.referencia, encoding="utf-8") as f:
        ref_crudo = " ".join(l.split(":", 1)[1] for l in f if ":" in l)
    puntos_ref = sum(ref_crudo.count(c) for c in ".?!")
    filas = []
    print(f"\nWhisper '{a.modelo}' en {dev}. Referencia: {len(ref)} palabras, {puntos_ref} puntos.\n")
    for nombre, prompt, hot, arrastre, modo in configs:
        t = time.time()
        if modo == "bloques":
            texto = "".join(w for _, _, w in tr.transcribir_por_bloques(asr, x, 5, hot, prompt, arrastre)).strip()
        else:
            segs, _ = asr.transcribe(x, language="es", beam_size=5, vad_filter=True, initial_prompt=prompt,
                                     hotwords=hot, condition_on_previous_text=arrastre)
            texto = " ".join(s.text.strip() for s in segs)
        variantes = [(nombre, texto)]
        corregido = tr.corregir(texto)
        if corregido != texto:
            variantes.append((nombre + " + correcciones", corregido))
        for etiqueta, txt in variantes:
            hip = ev.normalizar(txt)
            S, B, I, _ = ev.alinear([(w, "x") for w in ref], [(w, "x") for w in hip])
            wer = (S + B + I) / len(ref)
            res = [ev.evaluar_terminos(l, ref, hip) for l in listas]
            puntos = sum(txt.count(c) for c in ".?!")
            fila = [etiqueta, f"{wer:.1%}", I, f"{res[0][0]}/{res[0][1]} ({res[0][0] / res[0][1]:.0%})",
                    f"{res[1][0]}/{res[1][1]} ({res[1][0] / res[1][1]:.0%})", f"{time.time() - t:.0f} s", puntos]
            filas.append(fila)
            print(f"{fila[0]:46s} WER {fila[1]:>6} (inventadas {I:3d}) | vocabulario {fila[3]:>12} | "
                  f"nuevos {fila[4]:>11} | puntos {puntos}/{puntos_ref}")
            with open(os.path.join("transcripciones", "experimento_" + etiqueta[0] + ("_corr" if "+ corr" in etiqueta else "") + ".txt"),
                      "w", encoding="utf-8") as f:
                f.write(txt + "\n")
    with open("resultados_espanglish.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for fila in filas:
            w.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), a.audio, a.modelo] + fila)
    print("\nResultados agregados a resultados_espanglish.csv; textos en transcripciones\\experimento_*.txt")


if __name__ == "__main__":
    main()

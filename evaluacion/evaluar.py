#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara la transcripción del sistema contra una referencia escrita a mano.

Uso:
  python evaluacion\\evaluar.py evaluacion\\referencias\\prueba1.txt transcripciones/prueba1.txt --etiqueta "small, umbral 0.40"

Formato de ambos archivos (una línea por intervención; los tiempos son opcionales):
  [Uriel] (00:12 - 00:15): hola, ¿cómo están?
  [Ximena]: muy bien

Métricas:
  WER          porcentaje de palabras mal transcritas (sustituidas, borradas o insertadas). Menor es mejor.
  Atribución   de las palabras que sí se transcribieron bien, porcentaje con el hablante correcto. Mayor es mejor.
  Términos     (con --terminos lista.txt) porcentaje de términos de la lista, como "pull request" o
               "backend", que aparecen bien escritos. Se puede repetir para varias listas.
Cada evaluación se agrega a resultados.csv (y resultados_terminos.csv) para armar la tabla del reporte.
"""
import argparse
import csv
import datetime
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

PATRON = re.compile(r"^\s*\[([^\]]+)\]\s*(?:\([^)]*\))?\s*:?\s*(.*)$")
MARCA = re.compile(r"\[posible solapamiento[^\]]*\]", re.IGNORECASE)


# Deja el texto en minúsculas, sin acentos ni signos
def normalizar(texto):
    texto = MARCA.sub(" ", texto).lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")  # quita acentos
    texto = re.sub(r"[^\w\s]", " ", texto)
    return texto.split()


# Lee una transcripción y regresa cada palabra con su hablante
def leer(ruta):
    palabras = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            m = PATRON.match(linea)
            if m:
                hablante = m.group(1).strip().lower()
                palabras.extend((p, hablante) for p in normalizar(m.group(2)))
    return palabras


# Alinea dos textos palabra por palabra (distancia de edición)
def alinear(ref, hip):
    n, m = len(ref), len(hip)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        r, fila, previa = ref[i - 1][0], D[i], D[i - 1]
        for j in range(1, m + 1):
            fila[j] = min(previa[j] + 1, fila[j - 1] + 1, previa[j - 1] + (r != hip[j - 1][0]))
    i, j, S, B, I, aciertos = n, m, 0, 0, 0, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1][0] != hip[j - 1][0]):
            if ref[i - 1][0] == hip[j - 1][0]:
                aciertos.append((ref[i - 1][1], hip[j - 1][1]))
            else:
                S += 1
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            B, i = B + 1, i - 1
        else:
            I, j = I + 1, j - 1
    return S, B, I, aciertos


# Cuántas veces aparece un término en el texto
def contar(palabras, termino):
    n = len(termino)
    return sum(1 for i in range(len(palabras) - n + 1) if palabras[i:i + n] == termino)


# Porcentaje de términos técnicos bien escritos
def evaluar_terminos(ruta_lista, ref_palabras, hip_palabras):
    with open(ruta_lista, encoding="utf-8") as f:
        terminos = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    aciertos = total = 0
    fallos = []
    for t in terminos:
        tn = normalizar(t)
        if not tn:
            continue
        en_ref, en_hip = contar(ref_palabras, tn), contar(hip_palabras, tn)
        if en_ref == 0:
            continue
        aciertos += min(en_ref, en_hip)
        total += en_ref
        if en_hip < en_ref:
            fallos.append(f"{t} ({en_hip}/{en_ref})")
    return aciertos, total, fallos


# Compara una transcripción contra su referencia y guarda el resultado
def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Evalúa una transcripción contra su referencia")
    p.add_argument("referencia")
    p.add_argument("sistema")
    p.add_argument("--etiqueta", default="", help="qué configuración se probó (modelo, umbral, etc.)")
    p.add_argument("--csv", default="resultados.csv")
    p.add_argument("--terminos", action="append", default=[], help="lista de términos a revisar (repetible)")
    a = p.parse_args()

    ref, hip = leer(a.referencia), leer(a.sistema)
    if not ref:
        raise SystemExit("La referencia no tiene líneas con formato [Nombre]: texto")
    S, B, I, aciertos = alinear(ref, hip)
    wer = (S + B + I) / len(ref)
    bien = sum(r == h for r, h in aciertos)
    atrib = bien / len(aciertos) if aciertos else 0.0

    print(f"Palabras en la referencia: {len(ref)}  |  en el sistema: {len(hip)}")
    print(f"WER: {wer:.1%}   (sustituciones {S}, borradas {B}, insertadas {I})")
    print(f"Atribución de hablante: {atrib:.1%}   ({bien} de {len(aciertos)} palabras bien transcritas)")

    por_hablante = defaultdict(Counter)
    for r, h in aciertos:
        por_hablante[r][h] += 1
    print("\nPor hablante real -> lo que dijo el sistema:")
    for r, cont in sorted(por_hablante.items()):
        total = sum(cont.values())
        detalle = ", ".join(f"{h} {c / total:.0%}" for h, c in cont.most_common())
        print(f"  {r} ({total} palabras): {detalle}")

    nuevo = not os.path.exists(a.csv)
    with open(a.csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "etiqueta", "referencia", "sistema", "palabras_ref", "wer", "atribucion",
                        "sustituciones", "borradas", "insertadas"])
        w.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), a.etiqueta, a.referencia, a.sistema,
                    len(ref), f"{wer:.4f}", f"{atrib:.4f}", S, B, I])
    print(f"\nResultado agregado a {a.csv}")

    if a.terminos:
        ref_p, hip_p = [p for p, _ in ref], [p for p, _ in hip]
        nuevo_t = not os.path.exists("resultados_terminos.csv")
        with open("resultados_terminos.csv", "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if nuevo_t:
                w.writerow(["fecha", "etiqueta", "sistema", "lista", "aciertos", "total", "porcentaje"])
            for lista in a.terminos:
                ok, tot, fallos = evaluar_terminos(lista, ref_p, hip_p)
                pct = ok / tot if tot else 0.0
                print(f"\nTérminos de {lista}: {ok} de {tot} bien escritos ({pct:.1%})")
                if fallos:
                    print("  Fallaron: " + ", ".join(fallos))
                w.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), a.etiqueta, a.sistema, lista,
                            ok, tot, f"{pct:.4f}"])
        print("Términos agregados a resultados_terminos.csv")


if __name__ == "__main__":
    main()

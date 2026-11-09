# -*- coding: utf-8 -*-
"""
Opción "Traducir palabras en inglés": pasa al español las frases en inglés que dice la gente,
pero deja como están los términos técnicos (deploy, backend, commit...) y los nombres.

Cómo decide qué es inglés: idiomas.json trae qué tan común es cada palabra en inglés y en español
(escala zipf, de 0 a 8). Una palabra cuenta como inglés si es bastante más común en inglés que en
español. "is", "the", "anyway" sí; "no", "me", "real", "final" no, porque también son español.

El traductor es el modelo Marian opus-mt-en-es (Helsinki-NLP). Se busca en este orden:
  1. modelos/traductor/            versión convertida a CTranslate2 (rápida, sin internet)
  2. Helsinki-NLP/opus-mt-en-es    con la librería transformers (se descarga la primera vez)
Para dejar lista la opción 1 en la computadora del equipo: python preparar_traductor.py
"""
import json
import os
import re
import threading

AQUI = os.path.dirname(os.path.abspath(__file__))
CARPETA_MODELOS = "modelos"          # app.py la cambia a la carpeta del programa
MODELO_HF = "Helsinki-NLP/opus-mt-en-es"

DIFERENCIA = 1.2      # cuánto más común en inglés que en español tiene que ser una palabra
DIFERENCIA_SOLA = 2.0  # una palabra suelta entre español tiene que ser todavía más claramente inglés
MINIMO_EN = 2.5       # palabras rarísimas en inglés no cuentan (suelen ser errores de Whisper)

# muletillas que se dicen igual en español; traducirlas suena raro
SE_QUEDAN = {"ok", "okay", "wow", "bye", "hi", "hey", "bro", "yes", "please", "sorry", "full", "random", "crush"}

_lexico = None
_modelo = None
_candado = threading.Lock()


def _cargar_lexico():
    global _lexico
    if _lexico is None:
        for carpeta in (AQUI, os.getcwd()):
            ruta = os.path.join(carpeta, "idiomas.json")
            if os.path.exists(ruta):
                with open(ruta, encoding="utf-8") as f:
                    _lexico = json.load(f)
                break
        else:
            _lexico = {"en": {}, "es": {}}
    return _lexico


def _diferencia(palabra):
    """Qué tanto más común es la palabra en inglés que en español (positivo = suena a inglés)."""
    lex = _cargar_lexico()
    p = palabra.lower().strip("'")
    if re.search(r"[áéíóúñü¿¡]", p):
        return -9.0
    en = lex["en"].get(p, 0.0)
    if en < MINIMO_EN:
        return -9.0
    return en - lex["es"].get(p, 0.0)


def _protegidas(vocabulario):
    """Palabras que nunca se traducen: términos técnicos y nombres de las personas."""
    salida = set(SE_QUEDAN)
    if isinstance(vocabulario, str):
        vocabulario = [vocabulario]
    for frase in vocabulario or []:
        for p in re.findall(r"[\w']+", frase.lower()):
            salida.add(p)
    return salida


def tramos_en_ingles(texto, vocabulario=None):
    """[(inicio, fin)] de caracteres de `texto` que están en inglés y conviene traducir."""
    protegidas = _protegidas(vocabulario)
    fichas = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü']+", texto)]
    tipos = []
    for _, _, w in fichas:
        if w.lower() in protegidas:
            tipos.append("fijo")
        else:
            d = _diferencia(w)
            tipos.append("en" if d >= DIFERENCIA else ("neutra" if d > -1.0 else "es"))
    tramos, i = [], 0
    while i < len(fichas):
        if tipos[i] != "en":
            i += 1
            continue
        j = i
        # una racha de palabras en inglés; las neutras ("a", "real") y los términos técnicos entran
        # solo si después sigue más inglés ("I think the deploy is broken" es una sola frase)
        while j + 1 < len(fichas) and (tipos[j + 1] == "en" or
                                       (tipos[j + 1] in ("neutra", "fijo") and j + 2 < len(fichas)
                                        and tipos[j + 2] == "en")):
            # si entre dos palabras hay punto o coma fuerte, se respeta como frase aparte
            entre = texto[fichas[j][1]:fichas[j + 1][0]]
            if any(c in entre for c in ".?!;"):
                break
            j += 1
        sola = i == j
        if not sola or _diferencia(fichas[i][2]) >= DIFERENCIA_SOLA:
            tramos.append((fichas[i][0], fichas[j][1]))
        i = j + 1
    return tramos


# ------------------------------------------------------------------ el modelo
class _Marian:
    def __init__(self):
        ct2 = os.path.join(CARPETA_MODELOS, "traductor")
        if os.path.exists(os.path.join(ct2, "model.bin")):
            import ctranslate2
            import sentencepiece as spm
            self.modo = "ct2"
            self.tr = ctranslate2.Translator(ct2, device="cpu", compute_type="int8")
            self.sp_src = spm.SentencePieceProcessor(model_file=os.path.join(ct2, "source.spm"))
            self.sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(ct2, "target.spm"))
        else:
            from transformers import MarianMTModel, MarianTokenizer
            self.modo = "hf"
            self.tok = MarianTokenizer.from_pretrained(MODELO_HF)
            self.modelo = MarianMTModel.from_pretrained(MODELO_HF).eval()

    def traducir(self, frases):
        if not frases:
            return []
        if self.modo == "ct2":
            fuente = [self.sp_src.encode(f, out_type=str) + ["</s>"] for f in frases]
            res = self.tr.translate_batch(fuente, beam_size=2, max_decoding_length=128)
            return [self.sp_tgt.decode([t for t in r.hypotheses[0] if t != "</s>"]) for r in res]
        import torch
        with torch.inference_mode():
            lote = self.tok(frases, return_tensors="pt", padding=True)
            salida = self.modelo.generate(**lote, num_beams=2, max_new_tokens=128)
        return self.tok.batch_decode(salida, skip_special_tokens=True)


def cargar():
    """Carga el traductor (tarda unos segundos la primera vez). Lanza RuntimeError si no se puede."""
    global _modelo
    with _candado:
        if _modelo is None:
            try:
                _modelo = _Marian()
            except Exception as e:
                raise RuntimeError("No se pudo cargar el traductor de inglés a español. Revisa la conexión a "
                                   f"internet la primera vez, o corre preparar_traductor.py. ({e})")
    return _modelo


def disponible():
    return _modelo is not None


def traducir_texto(texto, vocabulario=None, traductor=None):
    """Regresa el texto con las partes en inglés traducidas al español."""
    tramos = tramos_en_ingles(texto, vocabulario)
    if not tramos:
        return texto
    traductor = traductor or cargar()
    protegidas = _protegidas(vocabulario)
    nuevas = traductor.traducir([texto[a:b] for a, b in tramos])
    salida, ultimo = [], 0
    for (a, b), nueva in zip(tramos, nuevas):
        original = texto[a:b]
        fijas = [w for w in re.findall(r"[\w']+", original) if w.lower() in protegidas]
        if any(w.lower() not in nueva.lower() for w in fijas):
            # el traductor cambió un término técnico: se traduce por pedazos, dejando el término igual
            nueva = _por_pedazos(original, fijas, traductor)
        salida += [texto[ultimo:a], _pulir(original, nueva)]
        ultimo = b
    salida.append(texto[ultimo:])
    return "".join(salida)


def _pulir(original, nueva):
    nueva = nueva.strip()
    if not original.rstrip().endswith(".") and nueva.endswith("."):
        nueva = nueva[:-1]
    if original[:1].islower() and nueva:
        nueva = nueva[0].lower() + nueva[1:]
    return nueva


def _por_pedazos(original, fijas, traductor):
    patron = r"\b(" + "|".join(re.escape(w) for w in fijas) + r")\b"
    partes = re.split(patron, original)
    huecos = [i for i, p in enumerate(partes) if p.strip() and p not in fijas]
    traducidas = traductor.traducir([partes[i].strip() for i in huecos])
    for i, t in zip(huecos, traducidas):
        partes[i] = (" " if partes[i].startswith(" ") else "") + _pulir(partes[i].strip(), t) + \
                    (" " if partes[i].endswith(" ") else "")
    return "".join(partes)

# Transcriptor multihablante en español (Opción A)

Escucha una conversación por el micrófono, identifica quién habla (entre las personas enroladas) y escribe la transcripción casi al momento en un TXT:

```
[Uriel] (00:12 - 00:15): Entonces la entrega es el viernes.
[Ximena] (00:15 - 00:18): Sí, pero primero hay que grabar las pruebas.
[Desconocido] (00:19 - 00:21): ¿Ya empezaron?
```

Cómo funciona: el detector de voz (Silero VAD) corta el audio en frases. Cada ~2 s se compara la huella de voz (ECAPA, SpeechBrain) contra la galería de personas enroladas. Whisper (faster-whisper) transcribe la frase y cada palabra se asigna a quien hablaba en ese momento. El retraso típico es de 1 a 3 s después de que alguien termina una frase. No se entrena ningún modelo.

## Instalación (una vez)

Requisitos: Python 3.10 a 3.12 (recomendado 3.11), un micrófono e internet la primera vez (descarga ~600 MB de modelos). No hace falta GPU.

Windows:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Mac / Linux: igual, pero se activa con `source .venv/bin/activate`.

## Uso

```
python transcriptor.py dispositivos                 # ver qué número tiene tu micrófono
python transcriptor.py enrolar --nombre Uriel       # cada persona, 40 s (repetible para sumar audio)
python transcriptor.py vivo                         # escuchar y transcribir; Ctrl+C para terminar
python transcriptor.py archivo pruebas/P01.wav      # procesar una grabación con el mismo flujo
python evaluar.py referencias/P01.txt transcripciones/P01_....txt --etiqueta "small, umbral auto"
```

- `vivo` guarda en `transcripciones/` el TXT y también el audio de la sesión (sirve para evaluar después).
- Si alguien manda su audio de enrolamiento grabado aparte, se copia a `voces/<Nombre>/` y se corre `python transcriptor.py galeria`.
- Al enrolar, el programa calcula un **umbral sugerido** comparando las voces entre sí y avisa si dos personas suenan muy parecido.

## Qué mover para mejorar (un cambio por prueba)

| Problema | Parámetro |
| --- | --- |
| Palabras mal transcritas | `--modelo medium` (más preciso, más lento) o `--beam 5` |
| Va muy atrasado | `--modelo base` o `--max-seg 4` |
| Pone "Desconocido" a gente enrolada | `--umbral` más bajo (ej. 0.30) o enrolar más audio |
| Confunde a dos personas | `--umbral` más alto y más audio de enrolamiento |
| Corta frases a la mitad | `--silencio-ms 700` |
| Detecta ruido como voz | `--vad 0.6` |
| Ver similitud y latencia por frase | `--detalle` |

## Protocolo de grabación del equipo

**Enrolamiento (cada persona):** 40-60 s con `enrolar`, leyendo el párrafo que muestra el programa y luego hablando libre. Mismo micrófono y misma distancia que en las pruebas. Lugar callado.

**Conversaciones de prueba** (2 min cada una, un solo micrófono al centro, guardar en `pruebas/`):

| Archivo | Situación |
| --- | --- |
| P01_2p_turnos | 2 personas, turnos claros |
| P02_3p_turnos | 3 personas, turnos claros |
| P03_4p_turnos | 4 personas, turnos claros |
| P04_3p_traslapes | 3 personas, interrupciones cortas a propósito |
| P05_3p_desconocido | 3 enrolados + 1 persona no enrolada |
| P06_3p_ruido | 3 personas con ruido de fondo (salón, cafetería) |

**Referencias:** por cada prueba, un TXT en `referencias/` con el mismo nombre, escrito a mano escuchando el audio, en el formato `[Nombre]: lo que dijo` (los tiempos no hacen falta). Los nombres deben ser iguales a los enrolados.

**Evaluación:** correr `archivo` + `evaluar.py` con cada configuración; cada resultado se agrega a `resultados.csv`, que es la tabla del reporte.

## Problemas comunes

- *No detecta el micrófono:* `python transcriptor.py dispositivos` y luego `--dispositivo N`.
- *Error de CUDA:* el programa cambia solo a CPU.
- *En Windows sale un error de "privilegio" o "symlink":* actualizar SpeechBrain (`pip install -U speechbrain`).
- *Muy lento en laptops sin GPU:* usar `--modelo base` para la demo y `small` o `medium` en modo `archivo` para la evaluación.

# Transcriptor multihablante en español (versión 2)

Escucha una conversación por el micrófono, identifica quién habla y escribe la transcripción casi al momento en un TXT:

```
[Uriel] (00:12 - 00:15): Entonces la entrega es el viernes.
[Ximena] (00:15 - 00:18): Sí, pero primero hay que grabar las pruebas.
[Desconocido 1] (00:19 - 00:21): ¿Ya empezaron?
```

Al terminar puede generar una **transcripción final** más precisa (`_final.txt`), procesando toda la grabación con su contexto completo y, si quieres, con un modelo más grande.

## Qué cambió en la versión 2

| Mejora | Qué hace |
| --- | --- |
| Separación de voces con Viterbi | Huellas de voz cada 0.25 s en ventanas solapadas de 1 s; el algoritmo solo cambia de hablante cuando la evidencia lo justifica, así los fragmentos cortos heredan el hablante correcto. |
| Desconocidos agrupados | Quien no está enrolado sale como `Desconocido 1`, `Desconocido 2`... según su voz. |
| Volumen nivelado | Las voces que llegan bajas (lejos del micrófono o por bocina) se amplifican antes de transcribir; el audio con volumen normal no se toca. |
| Nombres bien escritos | Los nombres enrolados y las palabras de `vocabulario.txt` se le pasan a Whisper. |
| Pasada final | `refinar` (o `vivo --final`) transcribe toda la sesión de una vez, con contexto completo y beam 5; pensada para usarse con `large-v3-turbo`. |
| Modo rápido automático | Si en vivo se juntan frases en cola, esa frase usa una separación más ligera para no atrasarse. |
| GPU automática | Si hay tarjeta NVIDIA con CUDA, usa `large-v3-turbo` en GPU (requiere `pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"`). |
| Menos falsos "solapamiento" | Solo se marca si el empate entre dos voces dura al menos 1 s (en un cambio de turno normal dura menos). |

### Resultados medidos (20 de octubre, Whisper `small`, CPU)
Banco de pruebas: 3 conversaciones armadas con frases reales de dos personas enroladas y una voz no enrolada, con 46 cambios de turno de 0.3 s o menos; y los párrafos leídos al enrolar como referencia de texto.

| Medida | Versión 1 | Versión 2 en vivo | Versión 2 pasada final |
| --- | --- | --- | --- |
| Palabras con el hablante correcto | 84.6 % | **97.4 %** | 96.4 % |
| Error de palabras (WER) en el párrafo leído | 6.9 % | 6.0 % | **5.7 %** |
| Voz detectada que sí queda transcrita | 86.8 % | **88.2 %** | 87.3 % |
| Voz por bocina o lejana transcrita | 79 % | **91 %** | 86 % |

Limitación que sigue: palabras sueltas justo en un cambio de turno pueden quedar con el hablante vecino. Para reducirlo, si una o dos palabras continúan una oración sin punto de por medio (como "Ni" + "tú ni Naomi..."), se quedan con el hablante del resto de la oración.

## Instalación (una vez)

Requisitos: Python 3.10 a 3.12 (recomendado 3.11), un micrófono e internet la primera vez.

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```
python probar_microfono.py                          # el micrófono sí capta sonido
python transcriptor.py enrolar --nombre Uriel       # cada persona, 40 s, SOLO habla ella
python transcriptor.py vivo                         # escuchar y transcribir; Ctrl+C para terminar
python transcriptor.py vivo --final                 # ...y al terminar genera la versión final
python transcriptor.py refinar transcripciones\sesion_XXXX.wav                       # versión final de una sesión
python transcriptor.py refinar transcripciones\sesion_XXXX.wav --modelo-final large-v3-turbo   # máxima precisión
python transcriptor.py archivo pruebas\P01.wav      # mismo flujo que en vivo, para evaluar
python evaluar.py referencias\P01.txt transcripciones\P01_....txt --etiqueta "v2 small"
```

### Vocabulario (opcional)
Crea un archivo `vocabulario.txt` en la carpeta del proyecto, con una palabra o nombre por línea (nombres de compañeros, del profesor, términos de la materia). Whisper tenderá a escribirlos bien.

## Modelos de Whisper: precisión contra velocidad

| Modelo | Descarga | Uso recomendado |
| --- | --- | --- |
| `small` | 0.5 GB | En vivo sin GPU (por defecto) |
| `medium` | 1.5 GB | Pasada final sin GPU si la computadora es potente |
| `large-v3-turbo` | 1.6 GB | En vivo con GPU (por defecto si hay GPU) y pasada final de máxima precisión |

En una computadora sin GPU, `large-v3-turbo` es lento para el modo en vivo, pero sirve muy bien para `refinar`, porque ahí no importa esperar.

### Usar la tarjeta NVIDIA (opcional, gran salto en precisión en vivo)
1. Revisa que tengas una: `nvidia-smi` debe mostrar tu tarjeta.
2. Instala las librerías de CUDA dentro del entorno:
   ```
   pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
   ```
3. Corre `python transcriptor.py vivo`. Debe decir `Whisper 'large-v3-turbo' listo en cuda`. Si falla, el programa vuelve solo a la CPU.

## Qué mover si algo no sale bien (un cambio por prueba)

| Problema | Qué probar |
| --- | --- |
| Palabras mal transcritas | `refinar` con `--modelo-final large-v3-turbo`; agregar nombres a `vocabulario.txt` |
| Va muy atrasado en vivo | `--modelo base` o `--max-seg 4` |
| Pone "Desconocido" a alguien enrolado | Enrolar más audio con el mismo micrófono; `--umbral` un poco más bajo |
| Confunde a dos personas | Enrolarlas de nuevo con más audio; `--umbral` un poco más alto |
| Corta frases a la mitad | `--silencio-ms 700` |
| Detecta ruido como voz | `--vad 0.6` |
| Ver similitud y latencia por frase | `--detalle` |

## Protocolo de grabación del equipo

**Enrolamiento (cada persona):** 40-60 s con `enrolar`, leyendo el párrafo y luego hablando libre. Mismo micrófono y distancia que en las pruebas, lugar callado. Solo habla la persona que se enrola.

**Conversaciones de prueba** (2 min cada una, un micrófono al centro, guardar en `pruebas/`):

| Archivo | Situación |
| --- | --- |
| P01_2p_turnos | 2 personas, turnos claros |
| P02_3p_turnos | 3 personas, turnos claros |
| P03_4p_turnos | 4 personas, turnos claros |
| P04_3p_traslapes | 3 personas, interrupciones cortas a propósito |
| P05_3p_desconocido | 3 enrolados + 1 persona no enrolada |
| P06_3p_ruido | 3 personas con ruido de fondo |

**Referencias:** por cada prueba, un TXT en `referencias/` escrito a mano: `[Nombre]: lo que dijo`. Los nombres deben ser iguales a los enrolados.

**Evaluación:** correr `archivo` (y `refinar`) + `evaluar.py` con cada configuración; cada resultado se agrega a `resultados.csv`.

## Problemas comunes

- *No detecta el micrófono:* `python transcriptor.py dispositivos` y luego `--dispositivo N`.
- *Parece congelado al iniciar:* la primera vez descarga modelos; espera.
- *Error de CUDA:* el programa cambia solo a CPU.
- *Avisos de "symlinks" o "HF_TOKEN":* son normales.

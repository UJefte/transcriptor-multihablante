# Transcriptor multihablante en español (versión 2.2)

Escucha una conversación por el micrófono, identifica quién habla y escribe la transcripción casi al momento en un TXT:

```
[Uriel] (00:12 - 00:15): Entonces la entrega es el viernes.
[Ximena] (00:15 - 00:18): Sí, pero primero hay que grabar las pruebas.
[Desconocido 1] (00:19 - 00:21): ¿Ya empezaron?
```

Al terminar puede generar una **transcripción final** más precisa (`_final.txt`), procesando toda la grabación con su contexto completo y, si quieres, con un modelo más grande.

## Qué cambió en la versión 2.2 (2 de noviembre)

Problema: con una sola persona hablando, partes de sus frases salían como «Desconocido».

- **Umbral por persona.** Antes había un solo umbral para todos (0.58), y lo subían mamá y Naomi porque sus voces se parecen entre sí (0.59). A Uriel, cuya voz es muy distinta a las demás (0.22), se le exigía lo mismo. Ahora cada persona tiene su propio umbral, calibrado con ventanas de 1 s (las mismas que usa la separación). Al abrir la app, una galería vieja se recalcula sola; en consola: `python transcriptor.py galeria`.
- **Las ventanas sin voz ya no cuentan.** Los tramos de silencio al final de cada frase no se parecen a nadie; ahora el detector de voz los descarta y ahí se queda quien venía hablando.
- **Frases en vivo de hasta 8 s, con pausa de 0.6 s para cerrarlas** (antes 6 s y 0.5 s): Whisper ve frases más completas.
- La app guarda sola la versión final en `transcripciones/<sesion>_final.txt`.

Medido con las grabaciones del 17, 20, 27 y 31 de octubre:

| | Antes | Ahora |
| --- | --- | --- |
| Tiempo atribuido a Uriel en 4 grabaciones donde solo habla él | 43 % a 91 % | 100 % en las 4 |
| Conversaciones armadas con audio que no se usó para la huella, 3 personas enroladas | 86.0 % | 89.9 % |
| Igual, con mamá sin enrolar (debe salir «Desconocido») | 69.0 % | 68.1 % |
| Igual, solo Uriel y Naomi | 98.2 % | 98.5 % |
| Error de palabras en vivo, lectura del 27 de octubre (Whisper `small`, CPU) | 16.7 % | 14.6 % |

Lo que sigue limitando: distinguir a una persona no enrolada de otra enrolada con voz parecida (mamá contra Naomi, 68 %), y las palabras sueltas o cantadas, que no tienen contexto para Whisper. La versión final siempre es más precisa que la de en vivo.

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

## Aplicación de escritorio (`app.py`)

Con el entorno activado:

```
python app.py
```

La primera vez tarda en abrir porque carga los modelos; el punto junto a «Modelos» se pone verde cuando están listos.

- **Conversación:** «Grabar» empieza a escuchar y la transcripción aparece en la ventana, cada persona con su color. Al detener se hace la pasada final (más precisa) y puedes cambiar entre «En vivo» y «Final». «Abrir audio…» transcribe un audio ya grabado; «Guardar TXT» exporta lo que está en pantalla. El audio de cada sesión se guarda en `transcripciones/`.
- **Personas:** quién está enrolado, «Volver a enrolar», «Borrar» y «Probar reconocimiento» (hablas 5 s y te dice a quién se parece tu voz, con la similitud de cada persona contra el umbral).
- **Enrolar voz:** enrolamiento guiado de 9 tomas con distintos tonos: lectura normal, preguntas, emoción, voz baja, voz fuerte, rápido, lento con números, términos técnicos y plática libre (unos 80 s de voz, el doble que antes).

Por qué el enrolamiento nuevo es más preciso:

1. La huella cubre cómo suena la persona en distintos tonos, no solo leyendo en tono plano, que es lo que pasa en una conversación real.
2. Cada toma se revisa al momento (volumen, saturación, segundos de voz, ruido de fondo) y se repite si sale mal.
3. Las tomas se comparan entre sí: si una no se parece a las demás (otra persona habló, hubo ruido), se marca y no se usa.
4. La voz nueva se compara con las ya enroladas: avisa si es casi igual a alguien (la misma persona con otro nombre) o si dos personas podrían confundirse.
5. La huella final descarta el 10 % de ventanas que menos se parecen al resto (toses, golpes, voces de fondo). Esto aplica también a `python transcriptor.py galeria`.

Para trabajar en la interfaz sin modelos ni micrófono (en cualquier PC): `python app.py --simulado`. La interfaz solo habla con `motor.py`; el motor real y el simulado tienen las mismas funciones.

Los comandos de consola de abajo siguen funcionando igual.

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

## Términos en inglés (español técnico de México)

En clase decimos "hice el deploy del backend" y no "hice el despliegue del servidor". Whisper sabe inglés, pero como le indicamos que el audio está en español, a veces escribe esos términos "a la española" ("el bakend", "un comit"). La versión 2.1 hace tres cosas:

| Qué | Archivo | Cómo ayuda |
| --- | --- | --- |
| Prompt en espanglish | (dentro del programa) | Whisper imita el estilo del texto previo: con frases técnicas mezcladas espera términos en inglés |
| Vocabulario técnico | `vocabulario_tecnico.txt` | Lista de términos que Whisper debe escribir en inglés. Pon primero los más importantes: solo caben unas 220 piezas de palabra |
| Correcciones | `correcciones.txt` | Reemplazos automáticos de errores que se repiten (`comit = commit`) |

Para comparar contra el comportamiento anterior se usa `--sin-espanglish`, que apaga las tres cosas.

### Experimento: ¿cuánto mejora?
1. Una persona lee en voz alta `pruebas/espanglish_lectura.txt` (unos 3 minutos) con `python transcriptor.py vivo`. Al terminar queda el audio en `transcripciones/`.
2. Transcribe ese mismo audio con y sin las mejoras:
   ```
   python transcriptor.py refinar transcripciones\sesion_XXXX.wav --modelo-final large-v3-turbo --salida transcripciones\esp_con.txt
   python transcriptor.py refinar transcripciones\sesion_XXXX.wav --modelo-final large-v3-turbo --sin-espanglish --salida transcripciones\esp_sin.txt
   ```
3. Mide las dos (en `referencias/espanglish.txt` cambia `[Uriel]` por el nombre enrolado de quien leyó):
   ```
   python evaluar.py referencias\espanglish.txt transcripciones\esp_sin.txt --etiqueta "sin espanglish" --terminos pruebas\terminos_en_vocabulario.txt --terminos pruebas\terminos_nuevos.txt
   python evaluar.py referencias\espanglish.txt transcripciones\esp_con.txt --etiqueta "con espanglish" --terminos pruebas\terminos_en_vocabulario.txt --terminos pruebas\terminos_nuevos.txt
   ```
4. Compara el porcentaje de términos bien escritos. `terminos_nuevos.txt` tiene términos que **no** están en el vocabulario técnico: muestra si la mejora sirve también para palabras que el programa no conocía.

Resultados (27 de octubre, una persona, 204 s, `large-v3-turbo` en GPU, `experimento_espanglish.py`; la referencia tiene 33 puntos):

| Configuración de la pasada final | Error general | Términos del vocabulario | Términos nuevos | Puntos |
| --- | --- | --- | --- | --- |
| Sin espanglish, todo el audio de una vez (como antes) | 10.3 % y 22.8 % en dos corridas | 83 % | 68 % | 40 y 51 |
| Espanglish arrastrando el texto anterior | 24.9 % | 80 % | 63 % | — |
| Espanglish sin arrastre + correcciones | 10.0 % | 89 % | 68 % | 5 |
| **Espanglish por bloques cortados en pausas (la que quedó)** | **8.7 %** | **89 %** | 68 % | 34 |

- Arrastrar el texto anterior entre ventanas hace que Whisper a veces entre en bucles ("en la tarde, en la tarde..."). Ni siquiera es estable: la configuración de antes dio 10.3 % en una corrida y 22.8 % en otra con el mismo audio.
- Sin arrastre, Whisper solo ve el prompt en los primeros 30 s y el resto sale sin puntos ni mayúsculas.
- Por bloques, el audio se corta en tramos de hasta 28 s, cada uno en la pausa más larga disponible, y cada tramo recibe el prompt completo. Además se quitan las repeticiones de 3 o más veces seguidas. Así no hay bucles, la puntuación se conserva en todo el audio y el error es el más bajo.
- Falta repetir la prueba con más lectores.

Para comparar configuraciones rápido sobre una grabación: `python experimento_espanglish.py transcripciones\sesion_XXXX.wav`

El texto tiene 438 palabras y 60 términos en inglés (89 apariciones). No usa números, porque Whisper podría escribir "500" donde se dijo "quinientos" y contaría como error.

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

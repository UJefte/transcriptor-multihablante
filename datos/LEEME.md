# Datos del proyecto

Datos que usamos para probar el transcriptor. No los usa el programa: el ejecutable
siempre empieza sin nadie enrolado y cada computadora enrola a sus propias personas.

## voces_enroladas/

Las tomas de enrolamiento de cada persona, tal como las graba la app (una carpeta por persona,
un .wav por toma: normal, pregunta, emoción, voz baja, voz fuerte, rápido, lento, técnico,
inglés y plática libre).

| Persona | Dónde se enroló | Tomas |
|---|---|---|
| Uriel J | PC de Uriel (con tarjeta NVIDIA) | 9 (versión anterior del enrolamiento, sin la toma en inglés) |
| Isabel | PC de Uriel | 10 |
| Geminis | PC de Uriel (voz de un asistente de IA, para probar una voz que no es de una persona del equipo) | 9 |
| Gera | Laptop de Gerardo (sin tarjeta NVIDIA) | 10 |
| Dulce | Laptop de Gerardo | 10 |
| Roman | Laptop de Gerardo | 10 |

Para usarlas en la app: copiar la carpeta de la persona a `voces/` y en Personas recalcular,
o correr `python transcriptor.py galeria`.

Parecido medido en la laptop de Gerardo: cada voz se parece 0.72 a sus propias tomas y 0.12 a
las de los demás (0.19 como máximo); umbrales entre 0.28 y 0.33.

## pruebas_grabadas/

Grabaciones de prueba con su transcripción en vivo (`.txt`), su versión final (`_final.txt`)
y el archivo de métricas que guarda la app (`metricas.csv`: tiempo de cada etapa, voces
encimadas encontradas y separadas, tiempo que habló cada persona).

- `pc_uriel/`
  - `sesion_20261112_170935`: plática por turnos entre Isabel y Uriel (75 s).
  - `sesion_20261112_172146`: plática con partes encimadas (48 s).
  - `sesion_20261112_173010`: Isabel y Uriel hablando al mismo tiempo (12 s); es el audio
    de la demostración de "Separar voces".
  - `metricas_personas.csv`: métricas por persona (botón "Medir" en Personas).
- `laptop_gerardo/`
  - `sesion_20261112_194339`: Dulce y Gera hablando al mismo tiempo (15 s).
  - `sesion_20261112_201927`: tres personas hablando al mismo tiempo (16 s); muestra el
    límite del separador, que solo separa a dos.
  - `diagnostico_laptop.txt`: datos de la laptop y de la galería.

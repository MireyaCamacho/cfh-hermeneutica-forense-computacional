# DOCUMENTO DE CAMBIOS — Tesis CFH
## Registro para actualización de capítulos (sesión 2026-07-03)

> Este documento lista cada cambio (valor viejo → valor nuevo, o decisión
> metodológica nueva) por capítulo. Se completa a medida que avanzan los
> pasos de cierre de Julián. Al final se aplica al borrador maestro.

---

## CAPÍTULO 4 — Metodología

### 4.x — Operacionalización del REP (y10_rep) [CAMBIO DE MÉTODO]

**Antes:** El REPExtractor aplicaba los 4 detectores (reconocimiento,
restitución, DIH, reparación) por igual a todos los corpus.

**Ahora:** Separación en tres ramas según el corpus, por diseño teórico:
- **Corpus A** (justicia ordinaria): solo ruptura epistémica institucional
  → restitución + DIH + reparación (el tribunal deja de eufemizar).
- **Corpus B** (autos JEP): todos los detectores → el auto recoge los
  reconocimientos de los comparecientes (justicia transicional) además del
  lenguaje institucional.
- **Corpus C** (habla oral): reconocimiento en 1ª persona + perdón +
  restitución, SIN el DIH técnico del tribunal.

**Además:** se ampliaron los patrones de detección para capturar el habla
ORAL de comparecientes ("yo asesiné", "le pido perdón,", "causamos daño",
"yo fui") que los patrones formales de autos no capturaban. Y se corrigió el
límite de longitud de spaCy (max_length) para procesar autos B de >1M chars.

**Justificación:** los patrones formales de autos JEP no capturaban el habla
oral; corrían los 4 detectores a todo por igual, contaminando la medición.
Regla de decisión REP auditable por corpus documentada.

### 4.x — Etiqueta MÁXIMO RESPONSABLE (MR) / NO-MR [VARIABLE NUEVA]

Se descartó rango_militar (no capturado sistemáticamente; solo en 3 de 5
inventarios). Se adoptó la categoría jurídica formal de la JEP: máximo
responsable (MR) vs no máximo responsable (no-MR).

**Regla de decisión (auditable):** imputado/determinado como máximo
responsable → MR; partícipe no determinante o sin imputación de MR → NO_MR;
evidencia ambigua → SIN_DATO. Fuente primaria: autos JEP (Corpus B);
corroboración: presentación del magistrado en audiencia (Corpus C).

**Distribución final:** 27 MR / 20 NO_MR (N=47).

### 4.x — Género discursivo del Corpus C [NOTA METODOLÓGICA NUEVA]

El Corpus C incluye dos géneros: audiencia de reconocimiento (Catatumbo,
Dabeiba, Huila, Costa Caribe) y versión voluntaria (Casanare/Torres,
2020-02-06, anterior a las audiencias). El CFH es sensible a esta diferencia:
el REP verbal bajo de la versión voluntaria refleja el género (relato/
descargo), no ausencia de reconocimiento. Esto muestra que el método se aplica
transversalmente a distintos formatos procesales.

---

## CAPÍTULO 5 — Resultados

### 5.x — Corrección del bug del canal verbal

El REPExtractor producía falsos negativos en Corpus C: comparecientes que
reconocían ("reconozco mi responsabilidad, pido perdón") puntuaban 0.000.
Ejemplo resuelto: José David Restrepo Solarte 0.000 → 0.118. Otros
recuperados: Guzmán 0.268, Yati 0.156, Samboni 0.434, Carvajal 0.312.

### 5.x — DIS/IEI por subcaso [VALORES RECALCULADOS]

Recalculados con el REP corregido (normalización z-score+sigmoid sobre
distribución conjunta A+B+C, N=1461). Hallazgos preservados:

| Subcaso     | DIS   | IEI   |
|-------------|-------|-------|
| Casanare    | 0.589 | 0.489 |
| Catatumbo   | 0.419 | 0.477 |  ← único IEI>DIS (disociación paradigmática)
| Costa Caribe| 0.510 | 0.474 |
| Dabeiba     | 0.491 | 0.449 |
| Huila       | 0.451 | 0.421 |

Tests: DIS A vs B n.s. (p=0.46); IEI A vs B p<0.001*** (A=0.528 vs B=0.462).
Catatumbo mantiene la disociación paradigmática (IEI>DIS). Huila fue el que
más se movió (y10 más alto=0.198; recuperó falsos negativos).

### 5.x — ICM tri-canal por compareciente

Regenerado con el REP corregido y corpus_type="C". Los 47 comparecientes con
tri-canal. REP verbal ahora con variación 0.06–0.91 sin saturación.

### 5.x — Análisis MR vs no-MR [RESULTADOS NUEVOS]

- ICM tri-canal: MR 0.399 vs no-MR 0.393, p=0.85 n.s.
- REP verbal: MR 0.179 vs no-MR 0.203, p=0.87 n.s.
- **Tiempo de palabra: MR mediana 1127s vs no-MR 413s, p=0.006, r=0.40 ***
  → la audiencia estratifica el espacio de reconocimiento por calidad jurídica.

La calidad de MR NO discrimina reconocimiento ni congruencia multimodal, pero
SÍ el tiempo de intervención.

### 5.x — Disociación multimodal [ANÁLISIS NUEVO]

Definida como sd de los 3 canales estandarizados (facial_z, vocal_z, verbal_z)
por compareciente. Independiente de la captura (rho~0 con tokens/facial/vocal)
y de la duración (rho=0.04). El canal VERBAL es la fuente de la disociación
(sd verbal disociados=0.318 vs congruentes=0.138).

Dos direcciones de incongruencia:
- Verbal alto / no-verbal bajo: Aguilera (verb 0.72/fac 0.04), Riveros
  (0.91/0.42), Castañeda (0.77/0.16).
- Verbal bajo / no-verbal alto: Calderón (0.10/fac 0.73), Contreras (0.10/0.77).

Por subcaso (solo audiencias de reconocimiento, sin versión voluntaria):
Kruskal-Wallis p=0.19 n.s. Huila (0.97) > Costa Caribe (0.51) p=0.048 (no
sobrevive corrección por comparaciones múltiples). MR vs no-MR p=0.50 n.s.

---

## CAPÍTULO 6 — Discusión / Conclusiones

### 6.x — Corrección: método de normalización

**Antes (texto tesis):** decía "percentil".
**Correcto:** z-score + sigmoid sobre distribución conjunta A+B+C.
Fórmula: sigmoid((x-μ)/σ) con sigmoid=1/(1+e^-x). Verificado en
code/normalizacion_definitiva_dis_iei.py.

### 6.x — Conclusiones del análisis del Corpus C

1. El reconocimiento no sigue la jerarquía de responsabilidad jurídica
   (MR/no-MR n.s. en reconocimiento y congruencia).
2. La audiencia estratifica el tiempo de palabra por calidad jurídica
   (MR ~3× más tiempo, p=0.006).
3. La (in)congruencia multimodal es un fenómeno individual, no explicado por
   rango, subcaso ni duración; el CFH la detecta donde el análisis unimodal no.
4. El CFH es sensible al género discursivo (versión voluntaria vs audiencia),
   aplicándose transversalmente a distintos formatos procesales.

---

## PENDIENTE DE COMPLETAR (pasos de Julián en curso)

- [ ] Paso 4 — IAA (κ): confirmar estado (κ=0.72 previo vs umbral >0.80).
- [ ] Paso 6 — análisis con/sin facial (robustez del ICM).
- [ ] Paso 7 — tabla de atrición/excluidos (existe icm_tricanal_final_excluidos.csv).
- [ ] Extraer valores ACTUALES del borrador maestro para contrastar viejo→nuevo
      (pendiente: localizar CFH_Tesis_BorradorMaestro_V2.docx).

# Conclusiones de la sesión — DIS/IEI tri-corpus + IAA (para ajustar capítulos)

Fecha: 2026-07-05
Autora: Mireya Camacho Celis — Tesis CFH, defensa agosto 2026

---

## 1. Decisión de diseño: fórmulas definitivas de los índices (OPCIÓN A)

Los tres índices sintéticos del framework CFH quedan así (los tres NO comparten
ningún indicador → dimensiones conceptualmente separadas):

- **DIS** (Injusticia Discursiva — superficie del lenguaje, Galtung):
  `DIS = 0.40·EBI_z + 0.30·SA_z + 0.30·(1−REP_z)`
  Mecanismos discursivos: eufemismo bélico (y1), supresión de agentividad (y2),
  y ausencia de ruptura epistémica positiva (1−y10). REP se ubica aquí porque
  fue anotado como marcador discursivo léxico-textual.

- **IEI** (Injusticia Epistémica — daño al sujeto como conocedor, Fricker):
  `IEI = 0.40·MAFAPO_z + 0.30·CIDH_z + 0.30·NV_z`
  Distancia semántica al léxico de víctimas (y8), a estándares DDHH (y9), y
  negación de victimización (y4). NV se ubica aquí porque negar a la víctima su
  condición es injusticia testimonial (credibility deficit de Fricker).

- **ICM** (Congruencia Multimodal, solo Corpus C): sin cambios.

**Justificación de la Opción A (IEI sin REP) sobre la Opción B (IEI con REP):**
doble respaldo. (1) Teórico: DIS e IEI no comparten indicadores → dimensiones
limpias. (2) Empírico: la correlación DIS–IEI es más baja en la Opción A
(ρ=0.164) que en la B (ρ=0.327); incluir REP en ambos generaba correlación
artificial.

## 2. Hallazgo empírico: el IEI captura la brecha epistémica justicia ordinaria vs JEP

Medias por corpus (normalización z-score+sigmoide conjunta A+B+C, n=946):

| Índice | A (ordinaria) | B (JEP escrito) | C (JEP oral) |
|--------|---------------|-----------------|--------------|
| DIS    | 0.510         | 0.498           | 0.394        |
| IEI    | 0.513         | 0.353           | 0.370        |

- **IEI:** A (0.513) >> B (0.353) ≈ C (0.370). La justicia ordinaria presenta
  mayor injusticia epistémica; la JEP (escrita y oral) la reduce ~0.15 puntos.
  Esto confirma la hipótesis central: los mecanismos transicionales corrigen la
  injusticia epistémica del archivo judicial ordinario.
- **DIS:** transversal entre A y B (0.510 vs 0.498); solo el registro oral (C)
  la reduce. La injusticia discursiva de *superficie* (gramática, eufemismo) es
  una convención del género jurídico compartida por ambos sistemas escritos.

**Interpretación teórica:** la corrección de la injusticia epistémica (adoptar
el léxico de las víctimas, dejar de negar su condición) y la de la injusticia
discursiva superficial (gramática institucional) son fenómenos DISOCIADOS. La
JEP corrige la primera pero conserva convenciones de la segunda.

## 3. Hallazgo metodológico: los tres índices miden dimensiones distintas

Matriz de correlaciones (Spearman):

| Par            | ρ           |
|----------------|-------------|
| DIS – IEI      | 0.164       |
| DIS – ICM      | −0.19       |
| IEI – ICM      | −0.11       |

Ninguna correlación supera 0.20 en valor absoluto. Los tres índices capturan
fenómenos independientes: discursivo (DIS), epistémico (IEI) y multimodal (ICM).
Esto valida empíricamente la arquitectura multi-dimensional del framework: no
son medidas redundantes de una misma "injusticia", sino tres dimensiones
separables. Es la evidencia cuantitativa del argumento dimensional del framework.

## 4. Fiabilidad inter-anotador (IAA) — dos rondas

- **Ronda 1 (anotación independiente, 100 fragmentos):**
  EBI κ=0.722, SA κ=0.617, NV κ=0.709 (los tres sustanciales);
  REP κ=0.309 (aceptable, el más bajo). Global macro = 0.589 (moderado).
- **Ronda 2 (revisión de los 34 desacuerdos de REP):** tras calibrar la
  definición operativa del constructo de Ruptura Epistémica Positiva, ambos
  anotadores revisaron los fragmentos en disputa. 28/34 convergieron; 6/34
  siguen en desacuerdo (casos genuinamente ambiguos, no forzados).
  REP κ=0.841 (casi perfecto). **Global macro = 0.722 (sustancial).**

El REP tuvo la menor concordancia inicial porque la frontera entre
reconocimiento genuino y lenguaje jurídico formulaico (p. ej. "homicidio en
persona protegida" en secciones RESUELVE) es interpretativamente difusa. La
calibración del constructo resolvió la mayoría de los desacuerdos.

## 5. Fortalecimiento del Corpus B

El Corpus B pasó de 54 secciones (cuyo texto segmentado no estaba persistido y
salía en 0.000 en todos los indicadores) a **80 secciones reales** segmentadas
de 14 documentos únicos por secciones temáticas (RECONOCIMIENTO 14,
HECHOS_Y_CONDUCTAS 14, CALIFICACION_JURIDICA 13, PATRONES_MACROCRIMINALES 11,
SANCIONES_PROPIAS 11, CONSIDERACIONES 8, RESUELVE 6, CUERPO 3). Los 6
indicadores (y1, y2, y4, y8, y9, y10) fueron recalculados sobre texto real.

## 6. Ajustes pendientes en el documento de tesis

- **Cap. 3 (Marco Teórico):** documentar la ubicación de REP en DIS (marcador
  discursivo) y NV en IEI (injusticia testimonial de Fricker).
- **Cap. 5 (Resultados):** reemplazar la tabla DIS/IEI con las medias nuevas
  (n=946); añadir la matriz de correlaciones DIS/IEI/ICM como evidencia de las
  tres dimensiones; reportar IAA de dos rondas (global 0.722).
- **Cap. 6 (Discusión):** integrar la disociación DIS–IEI (la JEP corrige lo
  epistémico pero no lo discursivo superficial) como hallazgo central; discutir
  que DIS es transversal al género jurídico escrito.

## 7. Pendiente técnico: SEM MG (multigrupo)

El DIS/IEI descriptivo está cerrado. Falta el **SEM multigrupo (MG-SEM)** que
modele formalmente la estructura entre corpus. Consideraciones para ese paso:
- Desbalance de unidades: A=819, B=80, C=47. Manejar con submuestreo de A o
  con estimador robusto.
- Con las unidades comparables y los índices ya construidos, el MG-SEM puede
  contrastar si la estructura de medición (cargas) es invariante entre corpus.

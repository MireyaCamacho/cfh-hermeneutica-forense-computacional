# CLAUDE.md — Instrucciones para agentes de IA

> Este archivo se lee automáticamente por Claude Code, Antigravity, Codex y otros agentes CLI. Contiene el contexto necesario para trabajar con este repo sin perder el rumbo.

---

## Identidad del proyecto

**Proyecto:** Hermenéutica Forense Computacional (CFH)
**Tipo:** Tesis de pregrado en Ciencia de Datos — Universidad Externado de Colombia
**Tesista:** Mireya Camacho Celis (Dra. en Derecho, Prof. UNAL)
**Director técnico:** Julián Zuluaga (Orbital Lab)
**Estado:** En desarrollo activo · Sprint 1 Cimientos

## Pregunta central

> **¿La justicia transicional colombiana (JEP) repara la violencia discursiva de la justicia ordinaria frente a las víctimas de falsos positivos, o la reproduce?**

Pregunta formal (versión de trabajo — pendiente cerrar con tesista):

> *¿En qué medida un framework computacional multimodal permite medir y caracterizar la injusticia discursiva en los archivos judiciales del Macrocaso 003 (JEP), cuantificando la distancia epistémica entre el discurso de la justicia ordinaria y la voz de las víctimas mediante indicadores textuales, prosódicos y de expresión facial?*

## Contexto histórico (CRÍTICO para entender el proyecto)

Mireya viene del derecho. Trabajó durante años analizando **manualmente** sentencias de falsos positivos en Excel, identificando textos aberrantes uno por uno. Llegó a ciencia de datos queriendo **automatizar y sofisticar** ese análisis cualitativo. Tiene:

- 6 informes PDF como antecedente cualitativo
- 15+ años investigando DDHH, justicia transicional y Macrocaso 003
- Póster presentado al departamento (año pasado) con la propuesta inicial
- Aplicación a congreso en Polonia (septiembre 2026) con la idea actualizada

Esto significa que el marco teórico (Habermas, Fraser, Galtung, Zehr) es **profundo y sólido**. Lo que necesita dirección es la parte computacional.

## Arquitectura — 3 capas + ICM

```
CAPA 1 — LÉXICA (texto)           "¿Cambian las palabras?"
  Indicadores: SA, NV, REP, CivDist, hedging, persona gramatical,
               léxico emocional, surprisal, encuadre narrativo
  Corpus: A (JO) + B (JEP escrita)

CAPA 2 — SEMÁNTICA (embeddings)   "¿Cambia el significado?"
  ConfliBERT-Spanish embeddings, distancia MAFAPO/CIDH,
  eco semántico (Cann 2025), UMAP/t-SNE
  Corpus: A + B + C

CAPA 3 — MULTIMODAL (voz+rostro)  "¿Es genuino?"
  Prosodia (eGeMAPS 88 params) + AUs faciales + head pose
  Corpus: C (audiencias JEP en video)

INTEGRACIÓN — ICM
  Índice de Congruencia Multimodal
  Mide alineación entre canales verbal y no-verbal
  ICM alto = reconocimiento genuino
  ICM bajo = reconocimiento performativo
```

### Decisiones arquitectónicas importantes

- **SEM de 4 factores fue descartado** — no ajusta con los datos (CFI=0.619, RMSEA=0.437). Se reemplaza por path analysis o PLS-SEM con factores simplificados.
- **Titans removido del título** — no tiene implementación estable con pesos públicos para español.
- **ICM debe ser temporal, no promediado** — ver sección "Temporalidad" abajo.

## Datos

### Corpus disponibles

| Corpus | Fuente | Tamaño | Estado | Path |
|--------|--------|--------|--------|------|
| **A** | Consejo de Estado + CSJ (2002-2008) | 819 secciones | Procesado, no en repo | `data/processed/corpus_a/` (gitignored) |
| **B** | Autos JEP Macrocaso 003 | 54 secciones (objetivo ≥200) | Insuficiente, ampliar | `data/processed/corpus_b/` |
| **C** | 5 audiencias JEP (Casanare, Catatumbo, Dabeiba, Huila, Costa Caribe) | 45.9h video, 1.17M chars | Transcrito con Whisper | `corpus_c/` |

Costa Caribe (La Popa) aún no procesada — 2 días de audiencia pegados en YouTube, requiere descarga segmentada.

### Polo normativo

**MAFAPO** (Madres de Falsos Positivos) + **CIDH** = referente semántico de "voz de las víctimas". Las distancias coseno a este polo son los indicadores y8 (MAFAPO) e y9 (CIDH).

## Stack tecnológico

| Componente | Herramienta | Estado |
|------------|-------------|--------|
| NLP base | ConfliBERT-Spanish (Yang et al., 2023) | Implementado |
| Fine-tuning | CFH-BERT v2 (IO tagging) | F1=0.584, 100 anotaciones |
| Embeddings | sentence-transformers | En requirements |
| ASR | Whisper large-v3 (medium usado actualmente) | Corpus C transcrito |
| Diarización | WhisperX / pyannote | Pendiente |
| Prosodia | OpenSMILE (eGeMAPS 88 params) | **No ejecutado — fallback 0.0** |
| Facial AUs | MediaPipe (actual) / OpenFace 3.0 (propuesto) | 4 audiencias procesadas |
| SEM | semopy | Descartado 4-factor; usar path analysis |
| Visualización | UMAP, t-SNE | 2 PNGs existentes |

## Estructura del repo (propuesta — en transición)

```
cfh/
├── README.md
├── CLAUDE.md                    ← Este archivo
├── pyproject.toml               ← Pendiente (no existe)
│
├── data/                        ← DATOS (gitignored excepto README.md)
│   ├── README.md                ← Inventario formal (pendiente)
│   ├── raw/                     ← Datos crudos
│   ├── processed/               ← Procesados
│   └── features/                ← Features por modalidad
│
├── experiments/                 ← EXPERIMENTOS numerados (pendiente)
│   └── EXP-XXX_nombre/
│       ├── notebook.ipynb
│       ├── config.yaml
│       ├── results.json
│       └── FINDINGS.md          ← Pregunta → Método → Resultado → Decisión
│
├── code/
│   └── src/
│       ├── ingestion/           ← Pipeline limpieza + segmentación
│       ├── features/            ← Extractores (SA, NV, REP, AUs, etc.)
│       └── ...
│
├── corpus_c/                    ← Transcripciones Whisper
├── outputs/                     ← Resultados generados
│   ├── capa3/                   ← AUs faciales + ICM
│   └── visualizaciones/
│
├── docs/                        ← DOCUMENTO de tesis
│   ├── estado_del_arte/
│   ├── sem_model/
│   └── capitulos/               ← (en textos/ como .docx)
│
├── tutoria/                     ← Material del director (Julián)
│   ├── guia-direccion-cfh.md/.pdf      ← Sesión 1
│   ├── sesion2-sprint1-cfh.md/.pdf     ← Sesión 2
│   └── image_*.png                     ← Infografías
│
├── .orbital/                    ← PROTOCOLO HANDOFF (Orbital Lab)
│   ├── config.yaml              ← Metadata del proyecto
│   ├── OBJETIVO.md              ← Sprint actual (qué hacer)
│   ├── README.md                ← Instrucciones + template ACTA
│   └── historial/               ← Actas archivadas
│
└── configs/                     ← YAML de configuración (no todos conectados al código)
```

## Ramas

| Rama | Uso | Quién |
|------|-----|-------|
| `main` | Producto final depurado | Mireya (autora) |
| `tutor` | Dirección técnica, protocolo .orbital | Julián (director) |
| `dev` | Rama de trabajo diario de Mireya | Mireya (pendiente crear) |

**Regla:** No pushear a `main` sin revisión del director. El merge `dev → main` pasa por `tutor` primero cuando Julián hace revisión.

## Protocolo HANDOFF (.orbital/)

El directorio `.orbital/` gestiona la comunicación entre director y tesista:

```
1. Director define .orbital/OBJETIVO.md (sprint actual)
2. Tesista trabaja → implementa, experimenta, documenta
3. Tesista entrega .orbital/ACTA_ENTREGA.md (resumen logros)
4. Director revisa, da feedback, define siguiente sprint
5. Acta se archiva en .orbital/historial/
```

Para detalles del formato ver `.orbital/README.md`.

## Convenciones de trabajo

### Cuando modifiques código

1. **Siempre trabaja en una rama** (no commitees directo a main)
2. **Un feature por commit** — mensajes descriptivos
3. **Tests primero** cuando agregues extractores de features
4. **Docstrings obligatorios** en módulos nuevos

### Cuando agregues features

Cada feature nuevo debe documentarse en la **tabla de features** (en `data/README.md` cuando exista):

| Feature | Modalidad | Corpus | Herramienta | Estado | Referencia |

### Cuando hagas experimentos

Crea `experiments/EXP-XXX_nombre/` con:
- `notebook.ipynb` — código ejecutable
- `config.yaml` — parámetros
- `results.json` — resultados numéricos (para comparar entre experimentos)
- `FINDINGS.md` — **el más importante**: pregunta → método → resultado → decisión que habilita

El `FINDINGS.md` es lo que alimenta el documento de tesis. Si no está escrito, el experimento no existe.

### Cuando tomes decisiones técnicas

Registra en `tutoria/decisiones.md` (crear si no existe):

```markdown
## DEC-XXX: Título de la decisión
- Fecha: YYYY-MM-DD
- Contexto: situación que llevó a decidir
- Decisión: qué se decidió
- Razón: por qué
- Alternativas consideradas
```

## Reglas de oro — lo que NO hacer

1. **NO promediar AUs o prosodia sobre audiencias completas** sin análisis temporal. Una expresión genuina es SOSTENIDA, no un flash.

2. **NO claim hallazgos con N=4 audiencias** como si fueran estadísticamente significativos. Son observaciones exploratorias.

3. **NO reportar resultados del ICM sin el canal vocal** implementado. Actualmente es bimodal (texto+facial), no trimodal.

4. **NO entrenar modelos deep learning** con 5-10 videos. Usar solo pre-trained + transfer learning.

5. **NO inventar referencias bibliográficas**. Si no la has verificado, no la cites.

6. **NO usar "reparación algorítmica" en título o pregunta** sin disclaimer. El framework mide, no repara. Hoffmann (2019) crítica esto explícitamente.

7. **NO hacer commit de datos crudos** (videos, audios, PDFs con copyright). Ir en `.gitignore`.

8. **NO duplicar archivos de anotaciones** con extensión `.json.json`. Consolidar versiones.

## Hallazgos actuales (validados y exploratorios)

### Validados estadísticamente

| Hallazgo | Soporte |
|----------|---------|
| Distancia MAFAPO (y8) A vs B significativa | p=0.0004 con ConfliBERT |
| Distancia CIDH (y9) A vs B significativa | p<0.001 |
| Brecha es estructural, no temporal | Persiste controlando periodo 2018-2023 |
| y3 (distancia léxica civil) predictor más fuerte de transición | β=-5.34, p<0.001 |
| `corpus_type` no agrega poder predictivo sobre indicadores léxicos | La diferencia es discursiva, no institucional |

### Exploratorios (no concluyentes)

- Patrón diferencial por rango militar (General 0.190 < Capitán 0.272 < Oficial 0.353)
  - **Solo 4 puntos de datos**, confounding total (magistrado, calidad video, etc.)
  - Tratar como observación, NO como resultado
- Ruptura epistémica JEP vs ordinaria en reparación (Mireya lo reporta en su documento)
- JEP reproduce eufemismos pero por citación de sentencias originales

## Papers semilla (lectura obligatoria)

1. **Gutiérrez-Osorio et al. (2025)** — [arXiv:2504.04325](https://arxiv.org/abs/2504.04325)
   *"Construyendo la verdad: minería de texto y redes lingüísticas en audiencias del Caso 03 de la JEP"*
   — Único paper con NLP sobre Macrocaso 003. Citación obligatoria.

2. **Baldivas et al. (2025)** — [MDPI Behavioral Sciences](https://www.mdpi.com/2076-328X/15/12/1707)
   *"LegalEye: Multimodal Court Deception Detection Across Multiple Languages"*
   — Hallazgo clave: en español, el texto pesa más que el video.

3. **Cann et al. (2025)** — [EPJ Data Science](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-025-00538-w)
   *"Semantic Echo"* — Método directamente aplicable para convergencia temporal.

Lista completa en `tutoria/sesion2-sprint1-cfh.md` y `Academy/Research/lines/02-cfh-hermeneutica-forense.md` (repo Orbital).

## Temas que requieren exploración (rabbit holes identificados)

Son potencialmente valiosos pero pueden absorber todo el tiempo. Evaluar, no ejecutar ciegamente:

- **Grafo ontológico de la JEP** — entidades (militares, víctimas, cortes) + relaciones → base de consulta contextual
- **Gemini Embedding 2** — encoder multimodal unificado (texto+audio+video en mismo espacio)
- **Sistema RAG de papers** — en desarrollo por Julián (pgvector + Supabase)
- **Agente del proyecto** — Mireya pendiente de proponer nombre

## Cómo trabajar con este repo (para agentes)

### Antes de empezar cualquier tarea

1. Leer `.orbital/OBJETIVO.md` — qué se espera del sprint actual
2. Leer el último archivo en `tutoria/` — material de la sesión más reciente
3. Verificar rama (`git branch --show-current`) — trabajar en la correcta
4. Si tarea viene de director → rama `tutor`; si es trabajo diario → `dev`

### Al escribir código

- Python 3.10+
- Type hints obligatorios en módulos nuevos
- Docstrings estilo Google (explicar POR QUÉ, no solo qué)
- Dataclasses para estructuras de datos
- Logging estructurado, no prints
- Tests en `code/tests/`

### Al generar resultados

- Resultados numéricos → `outputs/` con subdirectorio por capa
- Visualizaciones → `outputs/visualizaciones/` con nombre descriptivo
- **Siempre** crear `FINDINGS.md` explicando qué significa el resultado

### Al escribir documentación

- Capítulos de tesis → `docs/capitulos/` (consolidar los .docx de `textos/`)
- Decisiones técnicas → `tutoria/decisiones.md`
- Actas de entrega → `.orbital/ACTA_ENTREGA.md`, luego archivar

## Actores principales

- **Julián Zuluaga** (director técnico) — Orbital Lab, Universidad Externado
- **Mireya Camacho Celis** (tesista) — email: mireya.camacho@uexternado.edu.co
- **Ramel** (pendiente contacto) — referenciado en reunión, posible co-director o externo del departamento

## Referencias al ecosistema Orbital

Este proyecto está registrado como parte de la vertical **Academy** de Orbital Lab:

- **Supabase project_id:** `91a559d4-f770-4d5e-b111-0199334142de`
- **Línea de investigación:** `Academy/Research/lines/02-cfh-hermeneutica-forense.md` (repo orbital-os)
- **Protocolo HANDOFF:** `.orbital/` activo en rama `tutor`

---

*Actualizado: 2026-04-17*
*Versión: 1.0 — Inicialización del protocolo agentic*

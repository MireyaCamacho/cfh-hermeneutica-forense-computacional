# Hermenéutica Forense Computacional (CFH)

**Framework multimodal para medir la injusticia discursiva en justicia transicional colombiana**

---

**Tesis de pregrado · Ciencia de Datos**
**Autora:** Mireya Camacho Celis — *Dra. en Derecho (UNAL), Prof. de Ciencia Política*
**Director técnico:** Julián Zuluaga — *Orbital Lab*
**Institución:** Universidad Externado de Colombia
**Estado:** En desarrollo — Sprint 1 Cimientos · Abril 2026

---

## Resumen

Este proyecto propone el framework de **Hermenéutica Forense Computacional (CFH)** para medir empíricamente la injusticia discursiva en los archivos judiciales colombianos del fenómeno de los *falsos positivos* (Macrocaso 003 de la JEP).

Combina tres capas de análisis:

1. **Léxica** — indicadores textuales con NLP (ConfliBERT-Spanish)
2. **Semántica** — distancia a la voz de las víctimas (embeddings)
3. **Multimodal** — prosodia + expresión facial en audiencias orales

Las tres capas convergen en un **Índice de Congruencia Multimodal (ICM)** que operacionaliza el concepto de reconocimiento de Fraser: ¿el reconocimiento verbal en las audiencias JEP es **genuino** o **performativo**?

## Pregunta central

> ¿La justicia transicional colombiana (JEP) repara la violencia discursiva de la justicia ordinaria frente a las víctimas de falsos positivos, o la reproduce?

## Arquitectura

```
CAPA 1 — LÉXICA                   CAPA 2 — SEMÁNTICA              CAPA 3 — MULTIMODAL
(texto judicial)                  (embeddings)                    (audiencias)
      │                                 │                               │
      ▼                                 ▼                               ▼
 SA, NV, REP, CivDist              ConfliBERT                     eGeMAPS prosodia
 hedging, persona,                 distancia MAFAPO               AUs faciales
 léxico emocional,                 eco semántico                  head pose
 surprisal                         UMAP/t-SNE                     gaze
      │                                 │                               │
      └─────────────────────────────────┴───────────────────────────────┘
                                        │
                                        ▼
                    ÍNDICE DE CONGRUENCIA MULTIMODAL (ICM)
                    Genuino (congruente) vs Performativo (incongruente)
```

## Marco teórico

| Autor | Concepto | Rol |
|-------|----------|-----|
| **Habermas** (1981/1987) | Colonización sistémica del lenguaje | Base crítica |
| **Fraser** (1995, 2008) | Paridad participativa, reconocimiento | Operacionalización del reconocimiento |
| **Galtung** (1969, 1990) | Violencia cultural | Eufemismos como violencia simbólica |
| **Zehr** (2002) | Justicia restaurativa | Lo que la JEP debería lograr |

**Polo normativo:** MAFAPO (Madres de Falsos Positivos) + CIDH como referente de la voz de las víctimas.

## Corpus

| Corpus | Fuente | Tamaño | Estado |
|--------|--------|--------|--------|
| **A** | Consejo de Estado + CSJ (2002–2008) | 819 secciones | ✅ Procesado |
| **B** | Autos escritos JEP — Macrocaso 003 | 54 secciones | ⚠️ Ampliar a ≥200 |
| **C** | 5 audiencias JEP — 45.9h | 1.17M chars Whisper | ⚠️ Diarización pendiente |

## Stack tecnológico

- **NLP:** ConfliBERT-Spanish (Yang et al., 2023) · CFH-BERT v2 (IO tagging)
- **Embeddings:** sentence-transformers · distancia coseno MAFAPO/CIDH
- **ASR:** Whisper (medium/large-v3)
- **Prosodia:** OpenSMILE eGeMAPS (88 parámetros)
- **Computer Vision:** MediaPipe · OpenFace 3.0
- **SEM/Estadística:** semopy · path analysis · scipy
- **Visualización:** UMAP · t-SNE · matplotlib

## Estado del proyecto

| Componente | Estado |
|------------|--------|
| Marco teórico (Partes I–IV) | ✅ En revisión |
| Pipeline de ingesta (Corpus A/B) | ✅ Completo · 32 tests |
| Corpus C transcrito | ✅ Whisper, 5 audiencias |
| ConfliBERT embeddings (y8, y9) | ✅ Significativos p<0.001 |
| CFH-BERT v2 fine-tuning | ⚠️ F1=0.584 (100 anotaciones) |
| Capa 3 Facial (AUs) | ⚠️ 4 audiencias, sin análisis temporal |
| Capa 3 Vocal (prosodia) | ❌ Pendiente |
| ICM trimodal | ❌ Actualmente bimodal |
| SEM 4-factor | ❌ Descartado (CFI=0.619) |
| Path analysis simplificado | ⚠️ En diseño |
| Anotación con IAA | ❌ Pendiente 2º anotador |

## Hallazgos preliminares

### Validados estadísticamente

- **Brecha semántica estructural** entre justicia ordinaria y JEP (y8 p=0.0004, y9 p<0.001)
- **La brecha no es temporal** — persiste controlando periodo 2018–2023
- **y3 (distancia léxica civil)** es el predictor más fuerte de transición epistémica (β=-5.34)
- **El tipo institucional** no agrega poder predictivo sobre los indicadores léxicos — la diferencia es discursiva

### Exploratorios

- Patrón diferencial por rango militar (muestra pequeña, interpretar con cautela)
- JEP también reproduce eufemismos — por citación de sentencias originales, no como discurso propio

## Estructura del repositorio

```
cfh/
├── README.md                   Este archivo
├── CLAUDE.md                   Instrucciones para agentes IA
│
├── data/                       Datos (inventario pendiente)
├── experiments/                Experimentos numerados (EXP-XXX)
├── code/                       Paquete Python
│   └── src/
│       ├── ingestion/          Limpieza + segmentación
│       └── features/           Extractores léxicos, SEM, ICM
│
├── corpus_c/                   Transcripciones Whisper audiencias JEP
├── outputs/                    Resultados generados
│   ├── capa3/                  AUs faciales + ICM (4 audiencias)
│   └── visualizaciones/        UMAP, t-SNE
│
├── docs/                       Documento de tesis
│   ├── estado_del_arte/        Partes I–IV del marco teórico
│   └── sem_model/              Especificación del modelo
│
├── tutoria/                    Material de dirección técnica
│   ├── guia-direccion-cfh.pdf           Sesión 1 — Framework 3 capas
│   └── sesion2-sprint1-cfh.pdf          Sesión 2 — Sprint 1 Cimientos
│
├── .orbital/                   Protocolo HANDOFF de Orbital Lab
│   ├── OBJETIVO.md             Sprint actual
│   ├── config.yaml             Metadata del proyecto
│   ├── README.md               Instrucciones + template ACTA
│   └── historial/              Actas archivadas
│
└── configs/                    YAML de configuración
```

## Ramas

- `main` — producto final depurado (Mireya)
- `tutor` — dirección técnica, protocolo `.orbital/` (Julián)
- `dev` — trabajo diario de Mireya (pendiente crear)

## Protocolo de trabajo

Este proyecto usa el **protocolo HANDOFF** de Orbital Lab vía `.orbital/`:

1. Director define `.orbital/OBJETIVO.md` con los entregables del sprint
2. Tesista trabaja → implementa, experimenta, documenta
3. Tesista entrega `.orbital/ACTA_ENTREGA.md` al final del sprint
4. Director revisa, da feedback y define siguiente sprint
5. Actas se archivan en `.orbital/historial/` para trazabilidad completa

Ver `.orbital/README.md` para el template completo.

## Papers semilla

1. **Gutiérrez-Osorio et al. (2025)** — [arXiv:2504.04325](https://arxiv.org/abs/2504.04325) — JEP + NLP sobre Caso 03
2. **Baldivas et al. (2025)** — [LegalEye, MDPI](https://www.mdpi.com/2076-328X/15/12/1707) — Multimodal deception detection en español
3. **Cann et al. (2025)** — [Semantic Echo, EPJ](https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-025-00538-w) — Convergencia semántica temporal

## Referencias clave del marco teórico

- Yang, W. et al. (2023). ConfliBERT-Spanish. *IEEE CiSt*.
- Hu, Y. et al. (2022). ConfliBERT. *NAACL*.
- Kline, R.B. (2023). *Principles and Practice of SEM* (4th ed.).
- JEP (2021). Auto No. 019 — Macrocaso 003.
- Fraser, N. (1995). From redistribution to recognition. *NLR*.
- Galtung, J. (1990). Cultural violence. *JPR*, 27(3), 291–305.

## Contacto

**Tesista:** Mireya Camacho Celis
mireya.camacho@uexternado.edu.co

**Director técnico:** Julián Zuluaga
Orbital Lab · Universidad Externado de Colombia

---

*Universidad Externado de Colombia · Pregrado en Ciencia de Datos · 2026*

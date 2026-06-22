# CFH — Auditoría del Pipeline: Indicadores, Fuentes y Ubicaciones
**Fecha:** 21 de junio de 2026  
**Estado:** Auditoría pre-replicación. Base para pipeline ordenado.

---

## PARTE 1 — MAPA DE INDICADORES Y SU CONSTRUCCIÓN

### CAPA 1 — Indicadores léxicos (y₁ a y₄, y₁₀ a y₁₃)

| Indicador | Qué mide | Herramienta | Estado |
|-----------|----------|-------------|--------|
| **y₁ EBI** | Eufemismos bélico-institucionales (IO tagging) | CFH-BERT v2 | ⚠️ F1=0.0 — no operativo hasta IAA κ>0.80 |
| **y₂ SA** | Supresión de agentividad (pasivas, impersonales) | spaCy + dependency parsing | ✅ Calculado para A, B, C |
| **y₃ dist. léxica civil** | Distancia TF-IDF al léxico MAFAPO | TF-IDF + lexicón MAFAPO | ✅ Calculado para A, B, C |
| **y₄ NV** | Negación de victimización (IO tagging) | CFH-BERT v2 | ✅ F1=0.32 — usable con cautela |
| **y₁₀ REP** | Ruptura epistémica positiva (IO tagging) | CFH-BERT v2 | ✅ F1=0.77 — el más robusto |
| **y₁₁** | Registro léxico MAFAPO vs. militar | Beach et al. (2021) adaptado | ✅ Calculado Corpus C |
| **y₁₂** | Verbos atributivos | Beach et al. (2021) adaptado | ✅ Calculado A, B |
| **y₁₃** | Evidenciales | Beach et al. (2021) adaptado | ✅ Calculado A, B, C |

**Archivo fuente Capa 1:**
- `data/features/indicators_corpus_a.csv` — Corpus A (A-CE + A-CSJ)
- `data/features/indicators_corpus_b_v2.csv` — Corpus B (versión canónica, sin y₈/y₉)
- `data/features/indicators_corpus_c.csv` — Corpus C oral
- `data/features/capa1_nuevos_corpus_a.csv` — Indicadores adicionales Corpus A
- `data/features/capa1_nuevos_corpus_b.csv` — Indicadores adicionales Corpus B
- `data/indicators_y11_y12_y13.csv` — y₁₁/y₁₂/y₁₃ por subcaso

---

### CAPA 2 — Indicadores semánticos (y₇ a y₉)

| Indicador | Qué mide | Herramienta | Estado |
|-----------|----------|-------------|--------|
| **y₇ surprisal** | Imprevisibilidad léxica | BETO log-probabilidades | ⏳ No calculado definitivamente |
| **y₈ dist. MAFAPO** | Distancia coseno al centroide MAFAPO | ConfliBERT-Spanish + centroide v4 | ✅ En `indicators_completo_conflibert.csv` (col: `y8_mafapo_cs`) con centroide v3b. Centroide v4 disponible pero embeddings v4 no recalculados |
| **y₉ dist. CIDH** | Distancia coseno al centroide CIDH | ConfliBERT-Spanish + centroide v3 | ✅ En `indicators_completo_conflibert.csv` (col: `y9_cidh_cs`) |

**Centroides disponibles en `data/referencias/`:**
- `centroide_mafapo_v4.npy` — **CANÓNICO** (169 textos, ±7.7%, r=0.638) ← usar este
- `centroide_mafapo_v3b.npy` — versión anterior (67 textos, ±12.2%)
- `centroide_mafapo_v3.npy` — versión anterior
- `centroide_cidh_v3.npy` — **CANÓNICO** (25 fragmentos CIDH)

**Archivo fuente Capa 2:**
- `data/features/indicators_completo_conflibert.csv` — y₈/y₉ para Corpora A, B, C con centroide v3b
- `data/dis_iei_corpus_abc_definitivo.csv` — DIS e IEI calculados con normalización definitiva

**PENDIENTE:** Recalcular y₈/y₉ con centroide v4 sobre Corpus B (199 bloques). Los valores actuales usan v3b. La diferencia es pequeña pero debe documentarse.

---

### CAPA 3 — Indicadores multimodales (ICM)

| Indicador | Qué mide | Herramienta | Estado |
|-----------|----------|-------------|--------|
| **ICM facial** | Congruencia AUs (AU4 ceño vs AU12 sonrisa) | OpenFace 3.0 / MediaPipe | ✅ Calculado 4 subcasos (OpenFace) + Costa Caribe (MediaPipe N=32) |
| **ICM vocal** | Congruencia prosódica (shimmer, F0 std) | OpenSMILE eGeMAPS v02 | ✅ Calculado 4 subcasos, 10.897 ventanas |
| **ICM verbal v2** | Proximidad semántica a MAFAPO + léxico MAFAPO | ConfliBERT y₈ + y₁₁ | ✅ Calculado 5 subcasos |
| **ICM tri-canal v2** | 0.40×facial + 0.40×vocal + 0.20×verbal | Combinación ponderada | ✅ Calculado — valores canónicos en cfh.db |

**Archivos fuente Capa 3:**
- `aus_casanare_torres.csv` — AUs Casanare
- `aus_catatumbo_SPEAKER_01.csv`, `aus_catatumbo_SPEAKER_03.csv` — AUs Catatumbo
- `aus_dabeiba_SPEAKER_01.csv`, `aus_dabeiba_SPEAKER_03.csv` — AUs Dabeiba
- `aus_huila_SPEAKER_01.csv`, `aus_huila_SPEAKER_02.csv` — AUs Huila
- `indicators_corpus_c_capa1.csv` — Indicadores Capa 1 Corpus C
- `indicators_corpus_c_completo.csv` — Indicadores completos Corpus C
- `dis_iei_corpus_c.csv` — DIS e IEI por subcaso Corpus C

---

### ÍNDICES SINTÉTICOS

#### DIS Score (η₁ — Injusticia Discursiva)
```
DIS = 0.35×y₂_norm + 0.35×y₄_norm + 0.30×(1−y₁₀_norm)
```
- **Normalización:** percentil por corpus (no z-score — decisión metodológica)
- **Archivo canónico:** `data/dis_iei_corpus_abc_definitivo.csv` (col: `DIS`)
- **Script:** `code/normalizacion_definitiva_dis_iei.py`
- **Parsimonia:** 80/90 combinaciones de pesos ρ≥0.90 (89%) — `code/cfh_parsimonia_pesos_v2.py`

#### IEI (η₂ — Injusticia Epistémica)
```
IEI = 0.35×y₈_norm + 0.20×y₉_norm + 0.25×y₄_norm + 0.20×(1−y₁₀_norm)
```
- **Normalización:** percentil por corpus
- **Archivo canónico:** `data/dis_iei_corpus_abc_definitivo.csv` (col: `IEI`)
- **Script:** `code/normalizacion_definitiva_dis_iei.py`
- **Parsimonia:** 259/288 combinaciones ρ≥0.90 (90%)

#### ICM tri-canal v2
```
ICM_verbal_v2 = 0.60×ICM_verbal_v1(y₈) + 0.40×y₁₁_prop_MAFAPO
ICM_tri_v2 = 0.40×ICM_facial + 0.40×ICM_vocal + 0.20×ICM_verbal_v2
```
- **Valores canónicos en cfh.db:** tabla `capa3_icm_tri`
- **Scripts:** `code/icm_disociacion_v5b.py`, `code/icm_normalizado_v3.py`

---

## PARTE 2 — DÓNDE ESTÁ CADA COSA

### LOCAL (`C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional\`)

#### Resultados (data/)
| Archivo | Contenido | Estado |
|---------|-----------|--------|
| `data/features/indicators_corpus_a.csv` | y₁–y₁₃ Corpus A | ✅ |
| `data/features/indicators_corpus_b_v2.csv` | y₁–y₁₃ Corpus B (sin y₈/y₉) | ✅ versión canónica |
| `data/features/indicators_corpus_c.csv` | y₁–y₁₃ Corpus C | ✅ |
| `data/features/indicators_completo_conflibert.csv` | y₈/y₉ con centroide v3b para A+B+C | ✅ (v3b, no v4) |
| `data/dis_iei_corpus_abc_definitivo.csv` | DIS + IEI normalizados A+B+C | ✅ ARCHIVO MÁS COMPLETO |
| `data/dis_iei_corpus_c.csv` | DIS + IEI por subcaso C | ✅ |
| `data/indicators_y11_y12_y13.csv` | y₁₁/y₁₂/y₁₃ | ✅ |
| `data/referencias/centroide_mafapo_v4.npy` | Centroide MAFAPO v4 (canónico) | ✅ |
| `data/referencias/centroide_cidh_v3.npy` | Centroide CIDH v3 (canónico) | ✅ |
| `data/cfh.db` | Base de datos SQLite con Capa 3 | ✅ tablas capa3_* |

#### Scripts (code/)
| Script | Función | Estado |
|--------|---------|--------|
| `pipeline.py` | Pipeline principal | ✅ |
| `normalizacion_definitiva_dis_iei.py` | Calcula DIS e IEI con percentiles | ✅ SCRIPT CANÓNICO |
| `cfh_parsimonia_pesos_v2.py` | Análisis parsimonia pesos | ✅ |
| `cfh_ingesta_capa3.py` | Ingesta Capa 3 a cfh.db | ✅ |
| `cfh_muestreo_capa3_v2.py` | Guía muestreo 3 filtros | ✅ |
| `icm_disociacion_v5b.py` | ICM tri-canal v2 | ✅ versión canónica |
| `icm_normalizado_v3.py` | Normalización ICM | ✅ |
| `calcular_kappa_iaa.py` | Calcula κ IAA | ✅ listo para ejecutar |
| `calcular_aus_costa_caribe_v2.py` | AUs Costa Caribe | ✅ |
| `actualizar_dis_iei_costa_caribe.py` | Actualiza Costa Caribe en DIS/IEI | ✅ |

#### Notebooks (notebooks/)
| Notebook | Función | Estado |
|----------|---------|--------|
| `CFH_pipeline_centroide_v4_20jun2026.ipynb` | Pipeline centroide v4 + Whisper | ✅ último notebook Colab |

---

### GOOGLE DRIVE (`G:\Mi unidad\CHF_Corpus\`)

| Carpeta/Archivo | Contenido |
|-----------------|-----------|
| `corpus_c/` | 16 MP3 audiencias JEP + 11 TXT transcritos Whisper |
| `corpus_b/` | PDFs originales Corpus B (incluye `auto-RC-AI-016-2025-dabeiba.pdf`) |
| `referencias/centroide_mafapo_v4.npy` | Centroide MAFAPO v4 (copia Drive) |
| `referencias/centroide_cidh_v3.npy` | Centroide CIDH v3 (copia Drive) |
| `colab_corpus_b.tar.gz` | Tar con JSONs corpus B + features (subido hoy) |
| `data/features/indicators_corpus_b_v2.csv` | Copia Drive del CSV Corpus B |
| `data/features/indicators_completo_conflibert.csv` | y₈/y₉ con centroide v3b |

**Nota:** Hay duplicación de carpetas en Drive (`CFH_Hermeneutica_Forense_Computacional/` y `CHF_Corpus/`). Los archivos de resultados más actualizados están en el **local**, no en Drive.

---

### REPOSITORIO GIT (github.com/MireyaCamacho/cfh-hermeneutica-forense-computacional)

| Carpeta | Contenido |
|---------|-----------|
| `textos/` | Capítulos docx canónicos (Cap1-V5 a Cap6-V17 + Consolidada-v3) |
| `code/` | Scripts Python (ver lista arriba) |
| `notebooks/` | `CFH_pipeline_centroide_v4_20jun2026.ipynb` |
| `data/` | Solo metadatos y CSVs ligeros — los datos pesados NO están en git |
| `_md00_cfh_contexto_maestro_md.md` | Documento maestro actualizado |

**Lo que NO está en git (por tamaño):** PDFs originales, MP3/WAV, embeddings .npy grandes, cfh.db.

---

## PARTE 3 — ORDEN DE EJECUCIÓN PARA PIPELINE REPLICABLE

Este es el orden lógico para reproducir todos los resultados desde cero:

### FASE 0 — Preparación
```
1. Clonar repo
2. conda activate cfh
3. Copiar centroides a data/referencias/ (desde Drive)
4. Copiar cfh.db a data/ (desde Drive o local)
```

### FASE 1 — Capa 1 (léxica)
```
5. python code/pipeline.py --corpus A  → indicators_corpus_a.csv
6. python code/pipeline.py --corpus B  → indicators_corpus_b_v2.csv
7. python code/pipeline.py --corpus C  → indicators_corpus_c.csv
```

### FASE 2 — Capa 2 (semántica) ← REQUIERE GPU
```
8. [Colab] Cargar ConfliBERT-Spanish-Beto-Cased-v1
9. [Colab] Calcular embeddings corpus A+B+C con centroide v4
10. Guardar → indicators_completo_conflibert_v4.csv (PENDIENTE — actualmente existe v3b)
```

### FASE 3 — Índices DIS e IEI
```
11. python code/normalizacion_definitiva_dis_iei.py
    Input:  indicators_corpus_a.csv + indicators_corpus_b_v2.csv + indicators_completo_conflibert.csv
    Output: dis_iei_corpus_abc_definitivo.csv
```

### FASE 4 — Capa 3 (multimodal) ← REQUIERE GPU + VIDEO
```
12. [Colab] Whisper large-v3 sobre MP3 corpus C → TXT transcritos
13. [Colab] OpenSMILE eGeMAPS sobre WAV → features vocales
14. OpenFace 3.0 / MediaPipe sobre video → AUs faciales → aus_*.csv
15. python code/icm_disociacion_v5b.py → ICM tri-canal v2
16. python code/actualizar_dis_iei_costa_caribe.py → actualizar Costa Caribe
```

### FASE 5 — Ingesta a cfh.db y métricas de desempeño
```
17. python code/cfh_ingesta_capa3.py → capa3_icm_tri en cfh.db
18. python code/calcular_kappa_iaa.py → κ IAA (cuando llegue segundo anotador)
19. python code/cfh_parsimonia_pesos_v2.py → análisis sensibilidad pesos
```

---

## PARTE 4 — PENDIENTES CRÍTICOS

| Pendiente | Bloqueado por | Script listo |
|-----------|--------------|--------------|
| y₈/y₉ con centroide v4 para corpus B | GPU Colab + acceso textos | Parcialmente |
| α Cronbach definitivo | Tener y₈/y₉ v4 completos | `normalizacion_definitiva_dis_iei.py` |
| OCR Dabeiba 2025 | ✅ COMPLETADO HOY (1.447.696 chars) | `cfh_ocr_corpus_b.py` |
| IAA κ>0.80 | Segundo anotador | `calcular_kappa_iaa.py` |
| Nuevos videos corpus C | Identificar subcasos en playlist JEP | `cfh_muestreo_capa3_v2.py` |

---

## PARTE 5 — VALORES CANÓNICOS VERIFICADOS (no recalcular sin justificación)

| Métrica | Valor | Fuente | Archivo |
|---------|-------|--------|---------|
| y₈ MAFAPO Corpus B | 0.1913 | ConfliBERT v3b | `dis_iei_corpus_abc_definitivo.csv` |
| y₉ CIDH Corpus B | 0.2349 | ConfliBERT v3b | `dis_iei_corpus_abc_definitivo.csv` |
| ICM tri Catatumbo | 0.295 | OpenFace + eGeMAPS | `cfh.db:capa3_icm_tri` |
| ICM tri Casanare | 0.355 | OpenFace + eGeMAPS | `cfh.db:capa3_icm_tri` |
| ICM tri Huila | 0.421 | OpenFace + eGeMAPS | `cfh.db:capa3_icm_tri` |
| ICM tri Dabeiba | 0.490 | OpenFace + eGeMAPS | `cfh.db:capa3_icm_tri` |
| DIS Catatumbo | 0.110 | normalizacion_definitiva | `dis_iei_corpus_abc_definitivo.csv` |
| IEI Catatumbo | 0.624 | normalizacion_definitiva | `dis_iei_corpus_abc_definitivo.csv` |
| CFH-BERT v2 F1 macro | 0.58 | n=100 anotaciones | MLflow |

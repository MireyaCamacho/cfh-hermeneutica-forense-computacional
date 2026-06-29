# Protocolo de Reproducción — Framework CFH

**Hermenéutica Forense Computacional**
Reproducción de los índices DIS, IEI e ICM tri-canal sobre los corpus canónicos.

Mireya Camacho Celis · Universidad Externado de Colombia · 2026

---

## 1. Propósito

Este documento describe el procedimiento para reproducir, desde los datos crudos, los valores de los índices **DIS Score**, **IEI** e **ICM tri-canal** reportados en los capítulos 5 y 6 de la tesis. El protocolo está diseñado para ejecutarse sobre una copia limpia del repositorio, sin parches ni intervención manual, garantizando que cada cifra sea trazable y verificable.

> **Principio rector:** un solo pipeline, una sola segmentación, una sola normalización. Todos los indicadores del Corpus C se calculan sobre la misma base de **588 bloques de 2.000 caracteres**, evitando la unión de segmentaciones incompatibles.

---

## 2. Entorno

| Componente | Versión / especificación |
|---|---|
| Sistema operativo | Windows 11 |
| Python | 3.11 (entorno conda `cfh`) |
| Modelo de embeddings | `ConfliBERT-Spanish-BETO-Cased-v1` (eventdata-utd) |
| NLP sintáctico | spaCy con modelo de español |
| Librerías clave | pandas, numpy, scipy, torch, transformers |

---

## 3. Insumos requeridos

### 3.1 Transcripciones (Corpus C)

| Archivo | Subcaso | Tamaño |
|---|---|---|
| `corpus_c/catatumbo_audiencia_reconocimiento.txt` | Catatumbo | 117 KB |
| `corpus_c/costa_caribe.txt` | Costa Caribe | 258 KB |
| `corpus_c/casanare_torres.txt` | Casanare | 248 KB |
| `corpus_c/dabeiba_antioquia.txt` | Dabeiba | 287 KB |
| `corpus_c/huila.txt` | Huila | 267 KB |

> ⚠️ **Importante:** el Catatumbo canónico es el archivo de **117 KB** (`catatumbo_audiencia_reconocimiento.txt`), **NO** el de 18 KB (`catatumbo.txt`). Usar el incorrecto altera el número de bloques.

### 3.2 Indicadores precalculados

| Archivo | Contenido |
|---|---|
| `data/features/indicators_completo_conflibert.csv` | Corpus A+B (873 filas: A=819, B=54) |
| `data/features/indicators_corpus_c.csv` | y₈/y₉ del Corpus C (588 bloques, base `_b`) |

> **Nota sobre N_B=54:** el Corpus B canónico tiene **54 secciones** (universo de autos JEP del Macrocaso 003 disponibles). Versiones con 145 o 214 secciones corresponden a agregaciones descartadas y **no deben usarse**.

---

## 4. Pipeline de reproducción

Ejecutar los scripts en orden, desde la raíz del repositorio, con el entorno `cfh` activado.

### Paso 1 — Activar entorno

```bash
conda activate cfh
cd "...\CFH_Hermeneutica_Forense_Computacional"
```

### Paso 2 — Unificar el Corpus C y calcular DIS/IEI

Re-segmenta las transcripciones con el algoritmo canónico (2.000 caracteres), recalcula y₂/y₄/y₁₀ con los extractores reales, los une con y₈/y₉ por `bloque_id` (sin producto cartesiano) y calcula DIS/IEI con normalización z-score+sigmoide:

```bash
python cfh_unificar_corpus_c.py
```

**Verificaciones internas (asserts):** el script se detiene si el Corpus C supera 1.000 filas (previene el merge inflado), si hay `bloque_id` duplicados, o si la tasa de match cae por debajo de lo esperado. Si corre hasta el final, los datos son válidos.

Salidas:

```
data/indicators_corpus_c_unificado.csv   (588 bloques, 5 indicadores)
data/dis_iei_corpus_c_unificado.csv      (DIS/IEI por subcaso)
```

### Paso 3 — Verificar la robustez del DIS (A vs B)

Calcula el DIS bajo tres esquemas de normalización para confirmar que la no-significancia A vs B es propiedad de los datos, no del método:

```bash
python cfh_diag_dis_significancia.py
```

---

## 5. Resultados esperados

### 5.1 DIS e IEI por subcaso

| Subcaso | n | DIS | IEI | Patrón |
|---|---|---|---|---|
| Casanare | 124 | 0.592 | 0.525 | DIS>IEI |
| **Catatumbo** | 58 | **0.417** | **0.499** | **IEI>DIS** |
| Costa Caribe | 128 | 0.512 | 0.494 | DIS>IEI |
| Dabeiba | 144 | 0.491 | 0.479 | DIS>IEI |
| Huila | 134 | 0.467 | 0.463 | DIS>IEI |

Catatumbo es el único subcaso con **IEI>DIS**.

### 5.2 Tests A vs B

| Comparación | A | B | p | Sig. |
|---|---|---|---|---|
| DIS (z-score+sigmoide A+B+C) | 0.514 | 0.496 | 0.119 | n.s. |
| DIS (crudo, sin normalizar) | 0.668 | 0.655 | 0.279 | n.s. |
| **IEI** | 0.507 | 0.430 | <0.001 | *** |

El DIS agregado **no** separa A de B bajo ninguna normalización; el IEI **sí**. El DIS crudo (inmune a la normalización) confirma que es propiedad de los datos.

---

## 6. Archivos que NO deben usarse

Durante el desarrollo se generaron scripts y datos intermedios que producen resultados inflados o inconsistentes. **No** forman parte del pipeline y deben evitarse:

| Archivo / patrón | Problema |
|---|---|
| `normalizacion_definitiva_dis_iei.py` | Une segmentaciones `_c` y `_b` por nombre de subcaso → producto cartesiano (68.116 filas) |
| `actualizar_dis_iei_costa_caribe.py` | Generó la segmentación `_c` incompatible |
| `cfh_procesar_secciones_agregadas.py` | Infla el Corpus B a N_B≈214 con secciones agregadas |
| `cfh_cohend_v2.py` | Usa N_B=145 (no canónico) |
| `data/indicators_corpus_c_capa1_v2.csv` | Segmentación `_c` — incompatible con y₈/y₉ |
| `data/dis_iei_corpus_c_v3.csv` (7-jun) | CSV viejo, anterior a las correcciones |

> Estos archivos producen los valores inflados (DIS=0.808, N_B=214, Catatumbo Δ=0.514) que **NO** son los definitivos.

**Recomendación:** mover estos archivos a una carpeta `_deprecated/` fuera del pipeline, o documentarlos como históricos, para que ningún colaborador los ejecute por error.

---

## 7. Lista de verificación de entrega

- [ ] El Corpus C produce 588 bloques (no 68.116).
- [ ] El Corpus B usa N_B=54 (no 145 ni 214).
- [ ] El Catatumbo es el archivo de 117 KB.
- [ ] Los DIS/IEI coinciden con la tabla de la sección 5.1.
- [ ] El DIS A vs B da p=0.119 (n.s.) y el crudo p=0.279 (n.s.).
- [ ] El IEI A vs B da p<0.001 (***).
- [ ] Ningún script tiene rutas absolutas ni valores hardcodeados.
- [ ] Los asserts de `cfh_unificar_corpus_c.py` pasan sin error.

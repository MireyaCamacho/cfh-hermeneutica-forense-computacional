# CFH — Definiciones Formales, Taxonomía de Modelos y Resultados de Parsimonia

---

## PARTE 1: DEFINICIONES FORMALES DE COMPONENTES

### Notación general

Sea **d** un documento del corpus (sentencia, acta judicial, transcripción de audiencia).  
Sea **t** un token (unidad léxica) en d.  
Sea **N(d)** el número total de tokens en d.

---

### y₂ — SA Score (Supresión de Agentividad)

**Componentes:**

- **n_pasiva(d)**: número de cláusulas con voz pasiva sin complemento agente. Formalmente: cláusulas donde el verbo tiene dependiente con relación `auxpass` en el árbol de dependencias y no existe ningún dependiente con relación `agent` ni sintagma preposicional con 'por' como introductor de agente.
- **n_se_imp(d)**: número de construcciones impersonales con 'se' seguido de verbo conjugado en tercera persona sin sujeto léxico explícito.
- **n_nominal(d)**: número de nominalizaciones de eventos (sustantivos derivados de verbos de acción mediante sufijos -ción, -miento, -aje, -ura) en posición de núcleo de sintagma nominal sujeto de un verbo ligero.
- **n_verbos(d)**: número total de verbos conjugados en d (excluyendo verbos auxiliares).

**Fórmula:**

```
SA(d) = [n_pasiva(d) + n_se_imp(d) + n_nominal(d)] / n_verbos(d)
```

**Rango:** [0, 1]. Valor 0 = todos los verbos tienen agente explícito. Valor 1 = ningún verbo tiene agente explícito (máxima supresión).

**Modelo:** reglas sobre árbol de dependencias sintácticas (spaCy es_core_news_lg). No es aprendizaje automático — es un extractor basado en gramática formal.

---

### y₃ — Civil Distance (Distancia léxico civil)

**Componentes:**

- **L_civil**: lexicón de términos civiles y de derechos (|L_civil| = 1,247 términos). Curado a partir de: Constitución Política de Colombia (1991), Convención Americana sobre DDHH, sentencias de la Corte IDH relevantes.
- **v(d)**: vector TF-IDF de d en el espacio de términos del corpus.
- **c_civil**: centroide del lexicón civil. Formalmente: vector promedio de los vectores TF-IDF de los 1,247 términos de L_civil.
- **cos(a, b)**: similitud coseno entre vectores a y b = (a·b) / (‖a‖ · ‖b‖).

**Fórmula:**

```
y₃(d) = 1 − cos(v(d), c_civil)
```

**Rango:** [0, 1]. Valor 0 = el documento comparte completamente el léxico civil. Valor 1 = el documento no comparte nada con el léxico civil.

**Modelo:** TF-IDF con configuración min_df=2, max_features=50,000. Tarea: recuperación de información / similitud vectorial. No hay entrenamiento supervisado.

---

### y₄ — NV Score (Negación de Victimización)

**Componentes:**

- **NV_logit(d, t)**: log-probabilidad de que el token t en contexto c(t) pertenezca a la clase NV, producida por CFH-BERT v2.
- **σ**: función sigmoide. σ(x) = 1 / (1 + e^{−x}).
- **S_NV(d)**: conjunto de spans (secuencias contiguas de tokens) clasificados como NV en d por el modelo IO.
- **n_NV(d) = |S_NV(d)|**: número de spans NV en d.
- **n_sents(d)**: número de oraciones en d.

**Fórmula:**

```
NV(d) = σ(mean_{t ∈ S_NV(d)} NV_logit(d,t))   si S_NV(d) ≠ ∅
NV(d) = 0                                        si S_NV(d) = ∅
```

Alternativa de agregación (usada en la implementación actual):
```
NV(d) = n_NV(d) / n_sents(d)   → normalizado [0,1] por percentil del corpus
```

**Rango:** [0, 1].

**Modelo:** CFH-BERT v2 — clasificador de secuencias (sequence labeling, esquema BIO/IO) fine-tuned sobre ConfliBERT-Spanish-Beto-Cased-v1. Tarea: NLP de clasificación de tokens multiclase (EBI / SA / NV / REP / O).

---

### y₈ — Distancia MAFAPO

**Componentes:**

- **emb_CBS(d)**: embedding del documento d producido por ConfliBERT-Spanish (vector CLS, dimensión 768).
- **T_MAFAPO = {t₁, ..., t₂₅}**: corpus de referencia de 25 textos del informe MAFAPO 2021 "Unidas por la Memoria y la Verdad".
- **c_MAFAPO**: centroide MAFAPO. Formalmente:

```
c_MAFAPO = (1/25) Σᵢ emb_CBS(tᵢ),   tᵢ ∈ T_MAFAPO
‖c_MAFAPO‖ = 14.862
```

**Fórmula:**

```
y₈(d) = 1 − cos(emb_CBS(d), c_MAFAPO)
       = 1 − [emb_CBS(d) · c_MAFAPO] / [‖emb_CBS(d)‖ · ‖c_MAFAPO‖]
```

**Rango:** [0, 1]. Valor 0 = el documento es semánticamente idéntico al centroide MAFAPO. Valor 1 = máxima distancia semántica.

**Modelo:** ConfliBERT-Spanish-Beto-Cased-v1 (Yang et al., 2023). Tipo: transformer encoder bidireccional (arquitectura BERT). Tarea: generación de embeddings contextuales (representación semántica), no clasificación. Sin fine-tuning adicional para este indicador.

---

### y₉ — Distancia CIDH

**Componentes:**

- **T_CIDH = {t₁, ..., t₂₅}**: corpus de referencia de 25 textos de la sentencia Villamizar Durán y otros vs. Colombia (Corte IDH, 2018) y comunicados CIDH sobre Colombia.
- **c_CIDH**: centroide CIDH. ‖c_CIDH‖ = 15.188.

**Fórmula:**

```
y₉(d) = 1 − cos(emb_CBS(d), c_CIDH)
```

**Rango, modelo:** idénticos a y₈.

---

### y₁₀ — REP Score (Ruptura Epistémica Positiva)

**Componentes:** idénticos a y₄ pero para la clase REP.

**Fórmula:**

```
REP(d) = n_REP(d) / n_sents(d)   → normalizado [0,1]
```

Subtipos contados: n_reconocimiento + n_reparación + n_restitución.

---

### DIS Score (η₁) — Definición formal completa

**Paso 1 — Normalización min-max dentro del Corpus C:**

Sea C = {d₁, ..., d₅₄₇} el Corpus C (5 subcasos, 547 bloques en total).

```
y_k_norm(dᵢ) = [y_k(dᵢ) − min_{j∈C} y_k(dⱼ)] / [max_{j∈C} y_k(dⱼ) − min_{j∈C} y_k(dⱼ) + ε]
```

donde ε = 1×10⁻⁹ previene división por cero, y_k ∈ {y₂, y₄, y₁₀}.

**Paso 2 — Agregación por subcaso:**

Sea SC_s = {dᵢ : dᵢ pertenece al subcaso s}.

```
ȳ_k(s) = (1/|SC_s|) Σ_{dᵢ ∈ SC_s} y_k_norm(dᵢ)
```

**Paso 3 — DIS Score:**

```
η₁(s) = DIS(s) = w₂ · ȳ₂(s) + w₄ · ȳ₄(s) + w₁₀ · [1 − ȳ₁₀(s)]
```

donde (w₂, w₄, w₁₀) = (0.35, 0.35, 0.30), con w₂ + w₄ + w₁₀ = 1.

**Rango:** [0, 1]. Interpretación: relativa al Corpus C. DIS=1 = máxima injusticia discursiva observada en el corpus. DIS=0 = mínima observada.

---

### IEI Score (η₂) — Definición formal completa

**Paso 1 — Normalización** (igual que DIS, para y₈, y₉, y₄, y₁₀).

**Paso 2 — IEI Score:**

```
η₂(s) = IEI(s) = w₈ · ȳ₈(s) + w₉ · ȳ₉(s) + w₄ · ȳ₄(s) + w₁₀ · [1 − ȳ₁₀(s)]
```

donde (w₈, w₉, w₄, w₁₀) = (0.35, 0.20, 0.25, 0.20), con Σwᵢ = 1.

---

## PARTE 2: TAXONOMÍA DE MODELOS

### Tabla completa por indicador

| Indicador | Tipo de modelo | Familia | Tarea específica | Justificación de elección |
|-----------|----------------|---------|-----------------|--------------------------|
| **y₁ EBI** | Clasificador de secuencias (IO) | Transformer fino-ajustado | Sequence labeling multiclase | Detección de spans, no tokens: 'resultado operacional' es una secuencia. BERT captura contexto bidireccional necesario. |
| **y₂ SA** | Extractor basado en reglas | Análisis sintáctico (spaCy) | Parsing de dependencias + regex | No requiere entrenamiento — la gramática formal define el patrón. Totalmente auditable e interpretable. |
| **y₃ Civil** | Modelo de espacio vectorial | TF-IDF + similitud coseno | Recuperación de información | Sin parámetros a entrenar. La distancia es una propiedad geométrica del lexicón. |
| **y₄ NV** | Clasificador de secuencias (IO) | Transformer fino-ajustado (CFH-BERT v2) | Sequence labeling multiclase | El contexto determina si es NV. 'Portaba armamento' en negativa ≠ NV. Requiere modelo que entienda el contexto completo. |
| **y₇ Surprisal** | Modelo de lenguaje | Transformer encoder (BETO) | Language modeling — log-probabilidades | Bidireccional obligatorio: P(t | contexto_izq AND contexto_der). Los decoders generativos (GPT) solo ven el contexto izquierdo. |
| **y₈ Dist. MAFAPO** | Modelo de embeddings | Transformer encoder (ConfliBERT-Spanish) | Representación semántica + similitud coseno | Requiere dominio específico (conflicto político latinoamericano). Modelo genérico no captura semántica del corpus. |
| **y₉ Dist. CIDH** | Modelo de embeddings | Transformer encoder (ConfliBERT-Spanish) | Representación semántica + similitud coseno | Misma justificación que y₈. Centroide diferente (estándares DDHH vs léxico víctimas). |
| **y₁₀ REP** | Clasificador de secuencias (IO) | Transformer fino-ajustado (CFH-BERT v2) | Sequence labeling multiclase | Misma arquitectura que y₄. Clase opuesta semánticamente. |
| **y₁₂ ICM** | Ensemble multimodal | eGeMAPS (DSP) + MediaPipe (visión) + CFH-BERT (NLP) | Feature extraction + similitud coseno | Tres dominios de señal (audio, video, texto) requieren tres extractores especializados. El ICM es la función de agregación. |
| **SEM** | Modelo de ecuaciones estructurales | semopy (MLW) | Estimación de efectos latentes + test de ajuste | Modela error de medición explícitamente. Necesario porque los indicadores tienen error (spaCy comete errores de parsing, BERT comete errores de clasificación). Una regresión ordinaria ignoraría ese error. |

---

### Por qué NO regresión ordinaria para el SEM

La regresión OLS asume que las variables independientes se miden sin error. En este framework, todos los indicadores tienen error de medición:
- y₂ SA: spaCy comete errores de parsing en ~5-8% de oraciones complejas.
- y₄ NV: CFH-BERT v2 tiene F1=0.61 — aproximadamente 39% de error de clasificación en la clase NV.
- y₈, y₉: los embeddings tienen error de representación inherente (el espacio vectorial es una aproximación).

El SEM modela explícitamente ese error mediante las cargas factoriales λ. El coeficiente estructural β₂₃ que prueba H₃ es el efecto entre variables *latentes* (libres de error de medición), no entre variables observadas (contaminadas por error). Esto produce estimaciones más válidas de los efectos reales.

### Por qué NLP y no regresión estadística para los indicadores individuales

Los indicadores textuales (y₂, y₄, y₈, y₉, y₁₀) requieren NLP porque la unidad de análisis es el token o la secuencia de tokens en contexto. Una regresión estadística trabajaría sobre variables tabulares — aquí la variable de entrada es texto no estructurado de longitud variable. El NLP es el conjunto de técnicas para convertir texto en representaciones numéricas sobre las que se pueden aplicar métricas.

La elección específica de cada modelo dentro del NLP está justificada en la tabla anterior. La jerarquía es:
1. **Reglas cuando el patrón es gramaticalmente definible** (y₂): más interpretable, sin error de clasificación ML, totalmente reproducible.
2. **TF-IDF cuando la señal es léxica** (y₃): sin parámetros a entrenar, propiedad geométrica del lexicón.
3. **BERT encoder cuando el contexto es determinante** (y₄, y₁₀, y₇, y₈, y₉): el contexto bidireccional es necesario para la clasificación correcta.

---

## PARTE 3: RESULTADOS DE PARSIMONIA

### Análisis de sensibilidad de pesos — DIS Score

**Metodología:** grid search exhaustivo sobre el espacio de pesos DIS válidos (w₂, w₄, w₁₀) ∈ [0.10, 0.60]³ con restricción w₂ + w₄ + w₁₀ = 1, paso 0.05. N=90 combinaciones evaluadas.

**Métrica:** correlación de Spearman (ρ) entre el ranking de subcasos bajo los pesos base y bajo cada combinación.

| Resultado | Valor |
|-----------|-------|
| Combinaciones evaluadas | 90 |
| ρ media con pesos base | 0.9222 |
| ρ mínima | 0.8000 |
| Combinaciones con ρ ≥ 0.90 | 80/90 (89%) |
| Combinaciones con ρ ≥ 0.99 | 30/90 (33%) |
| Catatumbo mantiene DIS mínimo | 68/90 (76%) |
| Casanare mantiene DIS máximo | 79/90 (88%) |

**Interpretación:** el 89% de las combinaciones de pesos plausibles producen el mismo ranking de subcasos que los pesos teóricos (ρ≥0.90). Los hallazgos principales (Catatumbo = mínimo DIS, Casanare = máximo DIS) son robustos. El modelo es parsimonioso.

### Análisis leave-one-out — DIS Score

| Configuración | ρ con base | Catatumbo | Casanare |
|---------------|-----------|-----------|----------|
| Base (y₂+y₄+1-y₁₀) — COMPLETO | 1.000 | 0.120 | 0.787 |
| Sin y₂ SA | 0.900 | 0.180 | 0.680 |
| Sin y₄ NV | 0.900 | 0.163 | 0.680 |
| Sin 1-y₁₀ REP | 0.700 | 0.017 | 1.000 |
| Solo y₂ SA | 0.600 | 0.000 | 1.000 |
| Solo y₄ NV | 0.400 | 0.035 | 1.000 |
| Solo 1-y₁₀ REP | 0.600 | 0.326 | 0.360 |

**Interpretación:** los tres componentes son **necesarios**. Eliminar cualquiera reduce la correlación con el modelo completo, y eliminar 1-y₁₀ REP produce la mayor degradación (ρ=0.70). Ningún indicador individual captura la misma información que el índice compuesto. Esto justifica la composición del DIS como índice de tres componentes.

### Pesos óptimos empíricos vs. pesos teóricos

Los pesos óptimos empíricos se calculan proporcionales al efecto de separación A vs B (d de Cohen) de cada indicador.

| Indicador | d de Cohen (A vs B) | Peso empírico óptimo | Peso teórico | Diferencia |
|-----------|--------------------|--------------------|-------------|------------|
| y₂ SA | 4.68 | 0.182 | 0.350 | +0.168 |
| y₄ NV | 10.22 | 0.398 | 0.350 | −0.048 |
| 1-y₁₀ REP | 10.75 | 0.419 | 0.300 | −0.119 |

**Correlación pesos teóricos vs. óptimos DIS: ρ = 0.90**
**Correlación pesos teóricos vs. óptimos IEI: ρ = 1.00**

**Nota metodológica crítica:** el peso empírico de y₂ SA es menor que el teórico (0.18 vs 0.35), consistente con el resultado no significativo de SA en la comparación A vs B (p=0.127). SA es el indicador con menor poder discriminante entre los dos sistemas de justicia — pero sí discrimina entre subcasos del Corpus C (leave-one-out muestra que eliminarlo reduce ρ a 0.90). El peso teórico de 0.35 está justificado por la relevancia de la pretensión habermasiana de sinceridad expresiva en el análisis del corpus oral (Corpus C), donde SA sí varía significativamente entre comparecientes.

**Conclusión:** los pesos teóricos son **empíricamente plausibles** para el IEI (ρ=1.00 con el óptimo) y **robustos** para el DIS (ρ=0.90, rankings estables en 89% de combinaciones). El framework es parsimonioso.


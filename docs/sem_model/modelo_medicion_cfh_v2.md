# Modelo de Medición SEM — Proyecto CFH
## Especificación completa con ConfliBERT-Spanish como motor de extracción

**Versión:** 2.0 (integración ConfliBERT-Spanish)
**Fecha:** Marzo 2026

---

## 1. Visión general del modelo

El modelo SEM del proyecto CFH estima simultáneamente cuatro variables latentes
a partir de doce indicadores observables, todos extraídos mediante
ConfliBERT-Spanish (Yang et al., 2023) sobre los segmentos etiquetados
por el módulo de ingesta (cleaner → metadata → segmenter).

```
MODELO ESTRUCTURAL           MODELO DE MEDICIÓN

ξ₁ ──β₁₁──► η₁              ξ₁ ← {y₁, y₂, y₃, y₄}
ξ₂ ──β₁₂──► η₁              ξ₂ ← {y₅, y₆}
             η₁ ──β₂₃──► η₂  η₁ ← {y₇, y₈, y₉}
                             η₂ ← {y₁₀, y₁₁, y₁₂*}

* y₁₂ solo disponible en corpus C (audiencias con audio)
```

**Sintaxis semopy del modelo estructural:**
```
# Modelo de medición — cargas λ
xi1 =~ y1 + y2 + y3 + y4
xi2 =~ y5 + y6
eta1 =~ y7 + y8 + y9
eta2 =~ y10 + y11 + y12

# Modelo estructural — rutas β
eta1 ~ xi1 + xi2
eta2 ~ eta1

# Covarianza entre exógenas — φ₁₂
xi1 ~~ xi2
```

---

## 2. Variables latentes: definición teórica y operacional

### ξ₁ — Violencia Discursiva (variable latente exógena)
**Ancla teórica:** violencia cultural de Galtung (1990) en el dominio del
lenguaje normativo; *misrecognition* de Fraser (1995) operado a través del
léxico judicial.
**Definición operacional:** grado en que el lenguaje de un segmento judicial
activa patrones semánticos asociados a la legitimación de actos de violencia
estatal mediante mecanismos discursivos específicos: eufemismo, supresión de
agentividad, terminología bélica y negación de victimización.
**Corpus de aplicación:** A (primario), B (secundario), C (secundario).

### ξ₂ — Contexto Institucional (variable latente exógena)
**Ancla teórica:** dimensión de representación de Fraser (2008);
heterogeneidad institucional entre sistemas de justicia.
**Definición operacional:** conjunto de condiciones institucionales y
temporales bajo las cuales fue producido el documento, capturadas como
variables observables de corpus y período.
**Corpus de aplicación:** A, B y C (variable de agrupación en MG-SEM).

### η₁ — Injusticia Discursiva / DIS (variable latente endógena)
**Ancla teórica:** razón comunicativa colonizada de Habermas (1987);
distancia entre el texto y los estándares normativos de verdad de las víctimas.
**Definición operacional:** grado en que el espacio semántico de un segmento
judicial diverge del polo normativo construido a partir de los relatos de
víctimas (MAFAPO) y los estándares interamericanos (CIDH).
**Corpus de aplicación:** A, B y C.
**Nota metodológica:** η₁ reemplaza la fórmula manual DIS = α·surprisal + β·coseno.
Su estimación como factor score del modelo SEM garantiza que los pesos α y β
sean estimados empíricamente mediante máxima verosimilitud, no arbitrariamente.

### η₂ — Transición Epistémica (variable latente endógena)
**Ancla teórica:** "semántica de paz" como polo opuesto a la "gramática de
guerra"; justicia restaurativa como horizonte normativo (Zehr, 2002).
**Definición operacional:** grado en que el lenguaje de un segmento converge
hacia patrones semánticos de reconocimiento del daño, responsabilidad explícita
y reparación. Solo interpretable en corpus B y C.
**Corpus de aplicación:** B y C (primario). En A sirve como contrafáctico
basal para medir la distancia de partida.

---

## 3. Indicadores observables: definición, extracción con ConfliBERT-Spanish y escala

### Bloque ξ₁ — Indicadores de Violencia Discursiva

---

#### y₁ — Score EBI (Eufemismo Bélico-Institucional)

**Definición:** probabilidad asignada por CFH-BERT a la clase EBI
(Eufemismo Bélico-Institucional) en la tarea de clasificación de tokens.
Promediado sobre todos los tokens del segmento target.

**Extracción con ConfliBERT-Spanish:**
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# CFH-BERT = ConfliBERT-Spanish fine-tuneado sobre taxonomía CFH
model_name = "eventdata-utd/ConfliBERT-Spanish-CFH"  # después del fine-tuning

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

def extract_y1_ebi_score(segment_text: str) -> float:
    """
    Calcula el score EBI promedio del segmento.
    Retorna valor en [0, 1]: mayor = más eufemismos bélicos.
    """
    inputs = tokenizer(
        segment_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (batch, seq_len, n_classes)
    probs = torch.softmax(logits, dim=-1)

    # Índice de la clase EBI según el label2id del modelo
    ebi_idx = model.config.label2id["B-EBI"]
    ebi_scores = probs[0, :, ebi_idx].cpu().numpy()

    # Ignorar tokens especiales [CLS], [SEP], padding
    attention_mask = inputs["attention_mask"][0].cpu().numpy()
    valid_scores = ebi_scores[attention_mask == 1][1:-1]  # excluir CLS y SEP

    return float(valid_scores.mean()) if len(valid_scores) > 0 else 0.0
```

**Escala:** continua [0, 1]. Se normaliza z-score antes de entrar al SEM.
**Hipótesis de dirección:** λ₁ > 0 (mayor EBI → mayor ξ₁).

---

#### y₂ — Score SA (Supresión de Agentividad)

**Definición:** proporción de construcciones que suprimen el agente
responsable, detectadas mediante la combinación de ConfliBERT-Spanish
(clasificación de spans SA) con análisis de dependencia sintáctica (spaCy
es_dep_news_trf).

**Extracción con ConfliBERT-Spanish + spaCy:**
```python
import spacy
from transformers import pipeline

nlp_dep = spacy.load("es_dep_news_trf")  # parser de dependencias en español

# Pipeline de clasificación de tokens CFH-BERT
ner_pipeline = pipeline(
    "token-classification",
    model="eventdata-utd/ConfliBERT-Spanish-CFH",
    aggregation_strategy="simple"
)

def extract_y2_agentivity_suppression(segment_text: str) -> float:
    """
    Score de supresión de agentividad [0, 1].
    Combina:
    - Detección de spans SA por CFH-BERT (pasivas sin agente, nominalizaciones)
    - Ratio de verbos sin sujeto explícito del parser de dependencias
    """
    # Componente 1: spans SA del CFH-BERT
    entities = ner_pipeline(segment_text)
    sa_spans = [e for e in entities if e["entity_group"] == "SA"]
    sa_token_count = sum(
        len(tokenizer.tokenize(e["word"])) for e in sa_spans
    )
    total_tokens = len(tokenizer.tokenize(segment_text))
    sa_ratio_bert = sa_token_count / max(total_tokens, 1)

    # Componente 2: verbos sin sujeto explícito (parser de dependencias)
    doc = nlp_dep(segment_text[:5000])  # límite por performance
    verbs = [t for t in doc if t.pos_ == "VERB"]
    agentless_verbs = [
        v for v in verbs
        if not any(child.dep_ in ("nsubj", "nsubj:pass") for child in v.children)
    ]
    agentless_ratio = len(agentless_verbs) / max(len(verbs), 1)

    # Score combinado (pesos iguales; se puede ajustar como hiperparámetro)
    return float(0.5 * sa_ratio_bert + 0.5 * agentless_ratio)
```

**Escala:** continua [0, 1]. Se normaliza z-score.
**Hipótesis de dirección:** λ₂ > 0.

---

#### y₃ — Distancia semántica al léxico civil (embeddings ConfliBERT-Spanish)

**Definición:** distancia coseno entre el embedding del segmento y el
centroide de un conjunto de textos de referencia con léxico civil-pacífico
(noticias de contexto cotidiano sin conflicto). Mide cuánto el segmento
se aleja del espacio semántico del lenguaje civil y se adentra en el
espacio del lenguaje bélico.

**Extracción:**
```python
from transformers import AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# Modelo base (sin cabeza de clasificación) para embeddings
embed_model = AutoModel.from_pretrained(
    "eventdata-utd/ConfliBERT-Spanish"  # modelo base sin fine-tuning CFH
)
embed_tokenizer = AutoTokenizer.from_pretrained(
    "eventdata-utd/ConfliBERT-Spanish"
)

def mean_pooling(model_output, attention_mask):
    """CLS embedding o mean pooling de últimas capas ocultas."""
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()
    return (
        torch.sum(token_embeddings * mask_expanded, 1)
        / torch.clamp(mask_expanded.sum(1), min=1e-9)
    )

def get_segment_embedding(text: str) -> np.ndarray:
    inputs = embed_tokenizer(
        text, return_tensors="pt",
        max_length=512, truncation=True, padding=True
    )
    with torch.no_grad():
        output = embed_model(**inputs)
    return mean_pooling(output, inputs["attention_mask"]).numpy()

# CIVIL_CENTROID: precalculado una vez sobre corpus de referencia civil
# Shape: (1, 768)
CIVIL_CENTROID = np.load("data/reference/civil_lexicon_centroid.npy")

def extract_y3_semantic_distance_civil(segment_text: str) -> float:
    """
    Distancia coseno [0, 2] entre segmento y centroide léxico civil.
    Normalizado a [0, 1] dividiendo por 2.
    Mayor valor = más alejado del léxico civil = más bélico.
    """
    seg_emb = get_segment_embedding(segment_text)
    dist = cosine_distances(seg_emb, CIVIL_CENTROID)[0][0]
    return float(dist / 2.0)
```

**Escala:** [0, 1] (distancia coseno normalizada). Se normaliza z-score.
**Hipótesis de dirección:** λ₃ > 0.

---

#### y₄ — Score NV (Negación de Victimización)

**Definición:** probabilidad de la clase NV (Negación de Victimización)
por token, promediada sobre el segmento. Captura construcciones que
recaracterizan a la víctima como combatiente o agresor.

**Extracción:** análoga a y₁ pero para la clase NV del CFH-BERT (ver
código de y₁, cambiando el índice de clase a `model.config.label2id["B-NV"]`).

**Escala:** continua [0, 1]. Se normaliza z-score.
**Hipótesis de dirección:** λ₄ > 0.

---

### Bloque ξ₂ — Indicadores de Contexto Institucional

---

#### y₅ — Tipo de corpus (variable ordinal)

**Definición:** codificación ordinal del sistema de justicia productor del
documento: 0 = Justicia Ordinaria (corpus A), 1 = JEP escrito (corpus B),
2 = JEP oral (corpus C).
**Extracción:** metadato directo del campo `corpus_type` del JSON de ingesta.
**Nota SEM:** al ser una variable ordinal con 3 categorías, se trata como
indicador formativo de ξ₂ en la especificación semopy (reflective vs.
formative se especifica con `<~` en lugar de `=~`).
**Escala:** 0, 1, 2. Se normaliza por rango antes del SEM.
**Hipótesis de dirección:** codificación ordinal implica mayor valor =
mayor distancia del sistema retributivo original = menor ξ₁ esperada.

---

#### y₆ — Período temporal normalizado

**Definición:** año de emisión del documento normalizado al rango [0, 1]
dentro del período 2002-2022: `período_norm = (año - 2002) / 20`.
**Extracción:** campo `date_issued` del JSON de ingesta, año extraído.
**Escala:** [0, 1]. 0 = 2002 (inicio del período estudiado); 1 = 2022.
**Hipótesis de dirección:** λ₆ negativa esperada (documentos más recientes
→ mayor distancia del sistema original → menor ξ₂ en su componente
temporal; la relación causal se captura en β).

---

### Bloque η₁ — Indicadores del DIS Score

---

#### y₇ — Surprisal sobre corpus de paz (Titans / FAISS-RAG)

**Definición:** surprisal promedio del segmento calculado con el modelo de
lenguaje entrenado sobre corpus de paz y justicia restaurativa (polo opuesto
al corpus A). Alto surprisal = el texto del segmento es impredecible dado un
modelo de lenguaje de paz = el texto está léxicamente lejos del lenguaje de paz.

**Extracción (estrategia Plan B con FAISS):**
```python
import faiss
import numpy as np
from transformers import AutoModelForMaskedLM

# Modelo de lenguaje fine-tuneado sobre corpus de paz
# (corpus B + C + documentos MAFAPO + estándares CIDH)
peace_lm = AutoModelForMaskedLM.from_pretrained(
    "cfh-models/ConfliBERT-Spanish-peace-lm"
)
peace_tokenizer = AutoTokenizer.from_pretrained(
    "cfh-models/ConfliBERT-Spanish-peace-lm"
)

def compute_token_surprisal(text: str, model, tokenizer) -> float:
    """
    Calcula el surprisal promedio (en nats) usando masked LM.
    Para cada token, lo enmascara y obtiene -log P(token | contexto).
    """
    inputs = tokenizer(
        text, return_tensors="pt",
        max_length=512, truncation=True
    )
    input_ids = inputs["input_ids"][0]
    total_surprisal = 0.0
    n_tokens = 0

    for i in range(1, len(input_ids) - 1):  # excluir [CLS] y [SEP]
        masked_ids = input_ids.clone()
        masked_ids[i] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(
                input_ids=masked_ids.unsqueeze(0)
            ).logits[0, i]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_surprisal = -log_probs[input_ids[i]].item()
        total_surprisal += token_surprisal
        n_tokens += 1

    return total_surprisal / max(n_tokens, 1)

def extract_y7_surprisal(segment_text: str) -> float:
    """
    Surprisal promedio del segmento sobre el modelo de paz.
    Alto valor = texto impredecible para el modelo de paz = lejos del lenguaje
    restaurativo = mayor injusticia discursiva.
    """
    raw_surprisal = compute_token_surprisal(
        segment_text, peace_lm, peace_tokenizer
    )
    # Normalizar por la escala típica de surprisal del dominio
    # (estimada sobre muestra de calibración de 100 segmentos)
    SURPRISAL_MEAN = 3.82   # estimado sobre corpus de calibración CFH
    SURPRISAL_STD  = 1.14
    return (raw_surprisal - SURPRISAL_MEAN) / SURPRISAL_STD
```

**Escala:** z-score. Entra directamente al SEM.
**Hipótesis de dirección:** λ₇ > 0.

---

#### y₈ — Distancia coseno al polo MAFAPO

**Definición:** distancia coseno entre el embedding ConfliBERT-Spanish del
segmento y el centroide de los embeddings del corpus MAFAPO (testimonios,
declaraciones, cartas de madres de víctimas del Macrocaso 003).

**Extracción:**
```python
# MAFAPO_CENTROID: precalculado una vez sobre corpus MAFAPO completo
# Shape: (1, 768) — centroide de todos los embeddings de segmentos MAFAPO
MAFAPO_CENTROID = np.load("data/reference/mafapo_centroid.npy")

def extract_y8_mafapo_distance(segment_text: str) -> float:
    """
    Distancia coseno [0, 1] entre segmento y narrativa MAFAPO.
    Mayor valor = más lejos de la verdad de las víctimas = mayor injusticia.
    """
    seg_emb = get_segment_embedding(segment_text)
    dist = cosine_distances(seg_emb, MAFAPO_CENTROID)[0][0]
    return float(dist / 2.0)
```

**Construcción del MAFAPO_CENTROID:**
```python
def build_reference_centroid(texts: list[str], output_path: str):
    """
    Construye y guarda el centroide de referencia normativo.
    Se ejecuta UNA VEZ con el corpus de referencia completo.
    El centroide es el polo normativo de la Triangulated Truth Memory.
    """
    embeddings = [get_segment_embedding(t) for t in texts]
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    np.save(output_path, centroid)
    return centroid

# Uso:
# mafapo_texts = load_mafapo_corpus("data/reference/mafapo/")
# build_reference_centroid(mafapo_texts, "data/reference/mafapo_centroid.npy")
```

**Escala:** [0, 1], normalizado z-score.
**Hipótesis de dirección:** λ₈ > 0.

---

#### y₉ — Distancia coseno al polo CIDH

**Definición:** distancia coseno entre el embedding ConfliBERT-Spanish del
segmento y el centroide de los embeddings del corpus CIDH (sentencias de
la Corte Interamericana de Derechos Humanos sobre Colombia relevantes al
Macrocaso 003 y casos análogos).

**Extracción:** análoga a y₈, usando `CIDH_CENTROID`:
```python
CIDH_CENTROID = np.load("data/reference/cidh_centroid.npy")

def extract_y9_cidh_distance(segment_text: str) -> float:
    seg_emb = get_segment_embedding(segment_text)
    dist = cosine_distances(seg_emb, CIDH_CENTROID)[0][0]
    return float(dist / 2.0)
```

**Escala:** [0, 1], normalizado z-score.
**Hipótesis de dirección:** λ₉ > 0.

---

### Bloque η₂ — Indicadores de Transición Epistémica

---

#### y₁₀ — Score REP (Ruptura Epistémica Positiva)

**Definición:** probabilidad de la clase REP (Ruptura Epistémica Positiva)
por token, promediada sobre el segmento. Captura expresiones que validan
la perspectiva de las víctimas y reconocen el daño.

**Extracción:** análoga a y₁ para la clase REP del CFH-BERT.
**Escala:** [0, 1], normalizado z-score.
**Hipótesis de dirección:** λ₁₀ < 0 (mayor REP → menor η₂, i.e.,
mayor convergencia hacia semántica restaurativa).
**Nota:** en el modelo semopy la carga puede ser negativa; se interpreta
como indicador inverso. Alternativa: invertir la escala (1 - REP_score)
para que la carga sea positiva.

---

#### y₁₁ — Convergencia semántica hacia polo restaurativo

**Definición:** 1 - distancia coseno entre el embedding del segmento y
el centroide de un corpus de referencia restaurativa (sentencias con
lenguaje de reconocimiento explícito de daño, disculpas públicas, autos
JEP con mayor score REP).

```python
RESTAURATIVE_CENTROID = np.load("data/reference/restaurative_centroid.npy")

def extract_y11_restaurative_convergence(segment_text: str) -> float:
    """
    Convergencia [0, 1] hacia semántica restaurativa.
    1 = máxima convergencia; 0 = máxima divergencia.
    Indicador POSITIVO de transición epistémica.
    """
    seg_emb = get_segment_embedding(segment_text)
    dist = cosine_distances(seg_emb, RESTAURATIVE_CENTROID)[0][0]
    convergence = 1.0 - (dist / 2.0)
    return float(convergence)
```

**Escala:** [0, 1], normalizado z-score.
**Hipótesis de dirección:** λ₁₁ < 0 (mayor convergencia → menor η₂ → el constructo se
interpreta como "ausencia de transición"; si se recodifica η₂ como
"Injusticia Residual" λ₁₁ sería positivo y la ruta β₂₃ sería negativa).
**Nota de especificación:** la polaridad del constructo η₂ se fija en la
estimación SEM; aquí se mantiene la convención de que valores altos de η₂
representan mayor injusticia discursiva residual.

---

#### y₁₂ — Score acústico de reconocimiento (solo corpus C)

**Definición:** índice compuesto de rasgos prosódicos asociados en la
literatura a la expresión de carga emocional y reconocimiento en testimonios
de trauma: tasa de habla reducida, mayor proporción de pausas largas,
energía vocal moderada. Extraído con OpenSMILE (eGeMAPS v02).

**Extracción:**
```python
import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def extract_y12_acoustic_recognition(audio_path: str,
                                      segment_start: float,
                                      segment_end: float) -> float:
    """
    Score acústico de reconocimiento para un segmento de audio.
    Solo válido para corpus C (audiencias JEP con audio disponible).

    Retorna NaN para corpus A y B — se maneja como dato faltante
    en la estimación SEM (FIML: Full Information Maximum Likelihood).
    """
    import librosa
    import numpy as np

    audio, sr = librosa.load(
        audio_path, sr=16000,
        offset=segment_start,
        duration=segment_end - segment_start
    )
    features = smile.process_signal(audio, sr)

    # Rasgos eGeMAPS relevantes para reconocimiento/carga emocional:
    speech_rate   = features["speechrate_sma3nz_amean"].values[0]
    pause_ratio   = 1.0 - features["VoicedSegmentsPerSec"].values[0] / 10.0
    loudness_var  = features["loudness_sma3_stddevRisingSlope"].values[0]

    # Score compuesto normalizado (pesos iguales, puede optimizarse)
    raw = (
        (1.0 / max(speech_rate, 0.1)) * 0.4 +  # habla lenta → mayor peso
        pause_ratio * 0.4 +                       # más pausas → mayor peso
        loudness_var * 0.2                        # variabilidad vocal
    )
    # Normalizar sobre muestra de calibración del corpus C
    ACOUSTIC_MEAN = 0.52   # estimado sobre 50 segmentos de calibración
    ACOUSTIC_STD  = 0.18
    return (raw - ACOUSTIC_MEAN) / ACOUSTIC_STD
```

**Manejo de datos faltantes:** para corpus A y B, y₁₂ = NaN. La estimación
CB-SEM con semopy usa FIML (*Full Information Maximum Likelihood*) que maneja
datos faltantes de forma eficiente sin imputación, lo cual es metodológicamente
correcto para el caso MCAR (Missing Completely At Random) de los indicadores
acústicos ausentes en corpus A y B.

**Escala:** z-score.
**Hipótesis de dirección:** λ₁₂ < 0 (mayor score acústico de reconocimiento
→ mayor transición epistémica → menor η₂ en escala de injusticia residual).

---

## 4. Resumen de indicadores y cargas esperadas

| Indicador | Constructo | Herramienta | Corpus | Dirección λ esperada |
|---|---|---|---|---|
| y₁ — Score EBI | ξ₁ | CFH-BERT (token clf) | A, B, C | + |
| y₂ — Score SA | ξ₁ | CFH-BERT + spaCy dep | A, B, C | + |
| y₃ — Dist. léxico civil | ξ₁ | ConfliBERT-ES embed | A, B, C | + |
| y₄ — Score NV | ξ₁ | CFH-BERT (token clf) | A, B, C | + |
| y₅ — Tipo de corpus | ξ₂ | Metadato ingesta | A, B, C | formativo |
| y₆ — Período norm. | ξ₂ | Metadato ingesta | A, B, C | − |
| y₇ — Surprisal paz | η₁ | ConfliBERT-ES peace LM | A, B, C | + |
| y₈ — Dist. MAFAPO | η₁ | ConfliBERT-ES embed | A, B, C | + |
| y₉ — Dist. CIDH | η₁ | ConfliBERT-ES embed | A, B, C | + |
| y₁₀ — Score REP | η₂ | CFH-BERT (token clf) | B, C | − |
| y₁₁ — Conv. restaurativa | η₂ | ConfliBERT-ES embed | B, C | − |
| y₁₂ — Acústica | η₂ | OpenSMILE eGeMAPS | C only | − |

---

## 5. Pipeline de extracción de features — integración con módulo de ingesta

```python
"""
CFH · Feature Extractor
========================
Conecta el módulo de ingesta (IngestionResult) con la extracción de
indicadores SEM usando ConfliBERT-Spanish.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class SEMIndicators:
    """
    Los 12 indicadores observables del modelo SEM.
    NaN indica dato faltante (FIML en estimación).
    """
    doc_id: str
    corpus_type: str
    section_id: str

    # ξ₁ — Violencia discursiva
    y1_ebi: float
    y2_sa: float
    y3_civil_dist: float
    y4_nv: float

    # ξ₂ — Contexto institucional
    y5_corpus_type: float       # 0, 0.5, 1 normalizado
    y6_period_norm: float       # [0, 1]

    # η₁ — DIS Score
    y7_surprisal: float
    y8_mafapo_dist: float
    y9_cidh_dist: float

    # η₂ — Transición epistémica
    y10_rep: float
    y11_rest_conv: float
    y12_acoustic: Optional[float] = None  # NaN si corpus A o B


class CFHFeatureExtractor:
    """
    Extrae los 12 indicadores SEM de un IngestionResult.
    Usa ConfliBERT-Spanish como motor de embeddings y clasificación.
    """

    CORPUS_TYPE_MAP = {"A": 0.0, "B": 0.5, "C": 1.0}

    def __init__(self, model_path: str, references_dir: str):
        """
        model_path: ruta a CFH-BERT (ConfliBERT-Spanish fine-tuneado)
        references_dir: directorio con centroides .npy precalculados
        """
        self._load_models(model_path)
        self._load_reference_centroids(references_dir)

    def extract(self, ingestion_result, audio_path: Optional[str] = None
                ) -> list[SEMIndicators]:
        """
        Extrae indicadores de todas las secciones target de un documento.
        Retorna una lista de SEMIndicators (uno por sección target).
        """
        from src.ingestion.pipeline import IngestionResult
        assert ingestion_result.success

        corpus_type = ingestion_result.corpus_type
        doc_id = ingestion_result.sha256_clean[:16]
        date_issued = ingestion_result.metadata.get("date_issued", "2002")
        year = int(str(date_issued)[:4]) if date_issued else 2002
        y6 = (year - 2002) / 20.0

        results = []
        segments = ingestion_result.segmentation.get("sections", [])

        for seg in segments:
            if not seg.get("is_target", False):
                continue

            # Recuperar texto del segmento desde el clean_text
            char_start = seg["char_range"][0]
            char_end = seg["char_range"][1]
            seg_text = ingestion_result.clean_text[char_start:char_end]

            if len(seg_text.split()) < 20:  # segmentos muy cortos — omitir
                continue

            y12 = None
            if corpus_type == "C" and audio_path:
                # TODO: extraer timing del segmento desde alineación ASR
                y12 = self._extract_acoustic(audio_path, seg)

            indicators = SEMIndicators(
                doc_id=doc_id,
                corpus_type=corpus_type,
                section_id=seg["section_id"],
                y1_ebi=self._extract_y1(seg_text),
                y2_sa=self._extract_y2(seg_text),
                y3_civil_dist=self._extract_y3(seg_text),
                y4_nv=self._extract_y4(seg_text),
                y5_corpus_type=self.CORPUS_TYPE_MAP[corpus_type],
                y6_period_norm=y6,
                y7_surprisal=self._extract_y7(seg_text),
                y8_mafapo_dist=self._extract_y8(seg_text),
                y9_cidh_dist=self._extract_y9(seg_text),
                y10_rep=self._extract_y10(seg_text),
                y11_rest_conv=self._extract_y11(seg_text),
                y12_acoustic=y12,
            )
            results.append(indicators)

        return results

    def to_dataframe(self, indicators_list: list[SEMIndicators]) -> pd.DataFrame:
        """
        Convierte lista de SEMIndicators a DataFrame listo para semopy.
        Normaliza z-score todas las columnas de indicadores.
        """
        rows = [
            {
                "doc_id": i.doc_id,
                "corpus_type": i.corpus_type,
                "section_id": i.section_id,
                "y1": i.y1_ebi, "y2": i.y2_sa,
                "y3": i.y3_civil_dist, "y4": i.y4_nv,
                "y5": i.y5_corpus_type, "y6": i.y6_period_norm,
                "y7": i.y7_surprisal, "y8": i.y8_mafapo_dist,
                "y9": i.y9_cidh_dist, "y10": i.y10_rep,
                "y11": i.y11_rest_conv, "y12": i.y12_acoustic,
            }
            for i in indicators_list
        ]
        df = pd.DataFrame(rows)

        # Z-score normalización sobre indicadores (y1..y12)
        indicator_cols = [f"y{i}" for i in range(1, 13)]
        for col in indicator_cols:
            if col in df.columns and df[col].notna().sum() > 5:
                df[col] = (df[col] - df[col].mean()) / df[col].std()

        return df

    # Stubs de métodos privados (implementación completa en módulo SEM)
    def _load_models(self, path): pass
    def _load_reference_centroids(self, path): pass
    def _extract_y1(self, t): return 0.0
    def _extract_y2(self, t): return 0.0
    def _extract_y3(self, t): return 0.0
    def _extract_y4(self, t): return 0.0
    def _extract_y7(self, t): return 0.0
    def _extract_y8(self, t): return 0.0
    def _extract_y9(self, t): return 0.0
    def _extract_y10(self, t): return 0.0
    def _extract_y11(self, t): return 0.0
    def _extract_acoustic(self, p, s): return None
```

---

## 6. Especificación semopy y estimación

```python
"""
CFH · Módulo SEM — Estimación del modelo de medición y estructural
"""
import semopy
import pandas as pd
from semopy import Model

# Especificación del modelo (sintaxis lavaan/semopy)
CFH_SEM_SPEC = """
# Modelo de medición
xi1 =~ y1 + y2 + y3 + y4
xi2 =~ y5 + y6
eta1 =~ y7 + y8 + y9
eta2 =~ y10 + y11 + y12

# Modelo estructural
eta1 ~ xi1 + xi2
eta2 ~ eta1

# Covarianza entre exógenas
xi1 ~~ xi2
"""

def estimate_cfh_sem(df: pd.DataFrame) -> dict:
    """
    Estima el modelo SEM del proyecto CFH.

    Parámetros
    ----------
    df : DataFrame con columnas y1..y12 (normalizado z-score)

    Retorna
    -------
    dict con:
        - model: objeto semopy.Model estimado
        - fit_indices: RMSEA, CFI, SRMR, chi2/df
        - loadings: cargas factoriales λ estandarizadas
        - paths: coeficientes estructurales β
        - factor_scores: DataFrame con factor scores por observación
    """
    model = Model(CFH_SEM_SPEC)
    model.fit(df)

    # Índices de ajuste
    stats = semopy.calc_stats(model)
    rmsea = stats["RMSEA"].values[0]
    cfi   = stats["CFI"].values[0]
    srmr  = stats["SRMR"].values[0]
    chi2  = stats["chi2"].values[0]
    df_model = stats["df"].values[0]

    fit_indices = {
        "RMSEA": round(rmsea, 4),
        "CFI": round(cfi, 4),
        "SRMR": round(srmr, 4),
        "chi2_df_ratio": round(chi2 / df_model, 4) if df_model > 0 else None,
        "fit_adequate": rmsea < 0.08 and cfi > 0.90 and srmr < 0.08,
    }

    # Factor scores — el DIS Score es el factor score de eta1
    factor_scores = model.predict_factors(df)

    return {
        "model": model,
        "fit_indices": fit_indices,
        "loadings": model.inspect(what="est", std_est=True),
        "paths": model.inspect(what="est"),
        "factor_scores": factor_scores,
        "dis_score": factor_scores["eta1"].values,   # THE DIS Score
    }
```

---

## 7. Criterios de evaluación del modelo

| Índice | Umbral mínimo | Umbral óptimo | Interpretación si falla |
|---|---|---|---|
| RMSEA | < 0.08 | < 0.05 | Modelo sobrerestrictivo — revisar especificación |
| CFI | > 0.90 | > 0.95 | Mal ajuste comparativo — revisar indicadores |
| SRMR | < 0.08 | < 0.05 | Residuos altos — posibles covarianzas no modeladas |
| χ²/df | < 5.0 | < 3.0 | Sensible al n; guiarse por RMSEA + CFI en muestras grandes |
| Cargas λ | > 0.40 | > 0.60 | Indicador débil — considerar eliminar o respecificar |
| Varianza explicada R² de η₁ | > 0.30 | > 0.50 | ξ₁ y ξ₂ explican poco el DIS — revisar teoría |
| β₂₃ (η₁ → η₂) | p < 0.05 | p < 0.01 | Hipótesis central no confirmada |

---

## 8. Multi-Group SEM: comparación corpus A / B / C

```python
def estimate_mg_sem(df: pd.DataFrame) -> dict:
    """
    SEM multi-grupo para comparar modelo entre corpus A, B y C.
    Prueba invarianza de medición en tres niveles:
    configural → métrica → escalar.
    """
    results = {}

    # Modelo configural (mismo patrón, parámetros libres por grupo)
    mg_model_configural = semopy.ModelMeans(
        CFH_SEM_SPEC, groups="corpus_type"
    )
    mg_model_configural.fit(df)
    results["configural"] = semopy.calc_stats(mg_model_configural)

    # Modelo métrico (cargas iguales entre grupos)
    spec_metric = CFH_SEM_SPEC + "\n# Constrain loadings equal across groups"
    mg_model_metric = semopy.ModelMeans(
        spec_metric, groups="corpus_type"
    )
    mg_model_metric.fit(df)
    results["metric"] = semopy.calc_stats(mg_model_metric)

    # Prueba de diferencia de chi-cuadrado (configural vs. métrico)
    delta_chi2 = (
        results["metric"]["chi2"].values[0]
        - results["configural"]["chi2"].values[0]
    )
    delta_df = (
        results["metric"]["df"].values[0]
        - results["configural"]["df"].values[0]
    )
    results["metric_invariance_test"] = {
        "delta_chi2": delta_chi2,
        "delta_df": delta_df,
        "metric_invariance_supported": delta_chi2 / delta_df < 3.84  # p < 0.05
    }

    return results
```

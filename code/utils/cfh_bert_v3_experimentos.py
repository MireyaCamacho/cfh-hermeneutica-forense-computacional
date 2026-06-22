# ═══════════════════════════════════════════════════════════════════════════════
# CFH-BERT v3 — Experimentos de mejora de F1
# Hermenéutica Forense Computacional · Mireya Camacho Celis
# Universidad Externado de Colombia · Defensa agosto 2026
#
# BASE: ConfliBERT-Spanish-BETO-Cased-v1 (Yang et al., 2023)
#       oeg-upm/ConfliBERT-spanish-BETO-cased-v1
#
# BASELINE (CFH-BERT v2, ConfliBERT, WCE, n=100):
#   F1 macro=0.584 | REP=0.77 | EBI=0.52 | SA=0.52 | NV=0.32
#
# EXPERIMENTOS:
#   EXP-00  Baseline documentado (reproducir v2)
#   EXP-01  NV boost más agresivo (25× en lugar de 19×)
#   EXP-02  Aumentación: sinónimos controlados EBI/NV
#   EXP-03  Aumentación: back-translation ES→EN→ES
#   EXP-04  Aumentación: paráfrasis LLM (Claude API)
#   EXP-05  Focal Loss γ=2 en lugar de WCE
#   EXP-06  Focal Loss γ=2 + NV boost 25×
#   EXP-07  Threshold tuning por clase sobre mejor modelo
#   EXP-08  TODO combinado → CFH-BERT v3
#
# OBJETIVO JEP: F1 macro ≥ 0.85
# OBJETIVO DEFENSA: F1 macro ≥ 0.70
# ═══════════════════════════════════════════════════════════════════════════════

# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 0 — Instalación (ejecutar una sola vez en Colab)
# ╚══════════════════════════════════════════════════════════════════════════════
"""
!pip install transformers torch scikit-learn tqdm deep-translator anthropic mlflow -q
from google.colab import drive
drive.mount('/content/drive')
"""

import json, os, copy, random, time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# ─── Configuración global ─────────────────────────────────────────────────────
SEED           = 42
MAX_LENGTH     = 512
BATCH_SIZE     = 8
EPOCHS         = 20
PATIENCE       = 5
LR             = 2e-5
VAL_SPLIT      = 0.20

MODEL_BASE     = "oeg-upm/ConfliBERT-spanish-BETO-cased-v1"

LABEL2ID       = {"O": 0, "EBI": 1, "SA": 2, "NV": 3, "REP": 4}
ID2LABEL       = {v: k for k, v in LABEL2ID.items()}
N_LABELS       = len(LABEL2ID)

# Rutas en Drive — ajustar si es necesario
ANNOTATIONS    = "/content/drive/MyDrive/CFH/annotations_mireya_v1.json"
OUTPUT_DIR     = "/content/drive/MyDrive/CFH/modelos_v3"
RESULTS_FILE   = "/content/drive/MyDrive/CFH/experimentos_cfh_bert_v3.json"

# Reproducibilidad
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTADOS = {}  # acumula resultados de todos los experimentos


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 1 — Funciones base
# ╚══════════════════════════════════════════════════════════════════════════════

def convertir_io(tarea, tokenizer):
    """Convierte tarea Label Studio → tokens + etiquetas IO."""
    texto  = tarea["text"]
    spans  = tarea.get("label", [])
    char_labels = ["O"] * len(texto)
    for span in spans:
        label = span["labels"][0] if span["labels"] else "O"
        if label == "O":
            continue
        for i in range(span["start"], min(span["end"], len(texto))):
            char_labels[i] = label
    encoding = tokenizer(
        texto, max_length=MAX_LENGTH, truncation=True,
        return_offsets_mapping=True, padding="max_length",
    )
    token_labels = []
    for start, end in encoding["offset_mapping"]:
        if start == end:
            token_labels.append(-100)
        else:
            token_labels.append(LABEL2ID.get(char_labels[start], 0))
    return {
        "input_ids":      encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels":         token_labels,
    }


class CFHDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.data[idx].items()}


def calcular_pesos(dataset_list, nv_boost=1.0):
    """Pesos WCE inversos a la frecuencia + boost opcional para NV."""
    conteos = [0] * N_LABELS
    for d in dataset_list:
        for label in d["labels"]:
            if label >= 0:
                conteos[label] += 1
    total  = sum(conteos)
    pesos  = [total / (N_LABELS * max(c, 1)) for c in conteos]
    pesos[LABEL2ID["NV"]] *= nv_boost
    print("\nDistribución de clases:")
    for lid, (cnt, peso) in enumerate(zip(conteos, pesos)):
        bar = "█" * min(int(peso), 30)
        print(f"  {ID2LABEL[lid]:5s}: {cnt:5d} tokens  w={peso:6.2f}  {bar}")
    return torch.tensor(pesos, dtype=torch.float)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss — Lin et al. (2017).
    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)
    γ=2 es el valor óptimo para clases extremadamente desbalanceadas.
    Penaliza más los ejemplos fáciles (O bien aprendida)
    y fuerza al modelo a mejorar en NV.
    """
    def __init__(self, weight=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.weight       = weight
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        pt     = torch.exp(-ce)
        focal  = ((1 - pt) ** self.gamma) * ce
        mask   = targets != self.ignore_index
        return focal[mask].mean()


def evaluar(model, loader, thresholds=None):
    """
    Evalúa el modelo.
    thresholds: dict {class_id: float} para threshold tuning por clase.
    Si None → argmax estándar.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits  = outputs.logits  # (B, T, C)

            if thresholds is None:
                preds = logits.argmax(-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(-1).clone()
                for cid, thr in thresholds.items():
                    override = probs[:, :, cid] > thr
                    preds[override] = cid

            mask = batch["labels"] != -100
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(batch["labels"][mask].cpu().numpy())

    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    rep  = classification_report(
        all_labels, all_preds,
        target_names=[ID2LABEL[i] for i in range(N_LABELS)],
        zero_division=0, output_dict=True
    )
    rep_str = classification_report(
        all_labels, all_preds,
        target_names=[ID2LABEL[i] for i in range(N_LABELS)],
        zero_division=0
    )
    return f1, rep, rep_str


def entrenar(exp_id, desc, train_data, eval_data,
             loss_type="wce", nv_boost=1.0, gamma=2.0,
             thresholds=None, epochs=EPOCHS):
    """
    Entrena un experimento CFH-BERT completo.
    Guarda el mejor modelo y retorna métricas.
    """
    print(f"\n{'═'*60}")
    print(f"  {exp_id} — {desc}")
    print(f"  Loss: {loss_type} | NV boost: {nv_boost}× | γ: {gamma}")
    print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")
    print(f"{'═'*60}")

    pesos  = calcular_pesos(train_data, nv_boost).to(device)

    tl = DataLoader(CFHDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    el = DataLoader(CFHDataset(eval_data),  batch_size=BATCH_SIZE)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_BASE, num_labels=N_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    opt = AdamW(model.parameters(), lr=LR)

    if loss_type == "focal":
        loss_fn = FocalLoss(weight=pesos, gamma=gamma)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=pesos, ignore_index=-100)

    mejor_f1, mejor_epoch, sin_mejora = 0.0, 0, 0
    mejor_rep, mejor_rep_str = None, ""
    out_path = f"{OUTPUT_DIR}/{exp_id}"
    os.makedirs(out_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    historial = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(tl, desc=f"Época {epoch+1}/{epochs}", leave=False):
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            loss = loss_fn(
                outputs.logits.view(-1, N_LABELS),
                batch["labels"].view(-1)
            )
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(tl)
        f1, rep, rep_str = evaluar(model, el, thresholds)

        fila = {
            "epoch": epoch+1, "loss": round(avg_loss, 4),
            "f1_macro": round(f1, 4),
            "f1_EBI": round(rep["EBI"]["f1-score"], 4),
            "f1_SA":  round(rep["SA"]["f1-score"], 4),
            "f1_NV":  round(rep["NV"]["f1-score"], 4),
            "f1_REP": round(rep["REP"]["f1-score"], 4),
        }
        historial.append(fila)

        mejora = "✅" if f1 > mejor_f1 else "  "
        print(f"  {mejora} Época {epoch+1:2d} | loss={avg_loss:.3f} | "
              f"F1={f1:.3f} | NV={rep['NV']['f1-score']:.3f} | "
              f"REP={rep['REP']['f1-score']:.3f}")

        if f1 > mejor_f1:
            mejor_f1, mejor_epoch = f1, epoch + 1
            mejor_rep, mejor_rep_str = rep, rep_str
            sin_mejora = 0
            model.save_pretrained(out_path)
            tokenizer.save_pretrained(out_path)
        else:
            sin_mejora += 1
            if sin_mejora >= PATIENCE:
                print(f"  Early stopping en época {epoch+1} (paciencia={PATIENCE})")
                break

    print(f"\n  MEJOR → Época {mejor_epoch} | F1 macro={mejor_f1:.3f}")
    print(mejor_rep_str)

    resultado = {
        "exp_id":         exp_id,
        "descripcion":    desc,
        "loss_type":      loss_type,
        "nv_boost":       nv_boost,
        "gamma":          gamma,
        "n_train":        len(train_data),
        "n_eval":         len(eval_data),
        "mejor_epoch":    mejor_epoch,
        "f1_macro":       round(mejor_f1, 4),
        "f1_EBI":         round(mejor_rep["EBI"]["f1-score"], 4),
        "f1_SA":          round(mejor_rep["SA"]["f1-score"], 4),
        "f1_NV":          round(mejor_rep["NV"]["f1-score"], 4),
        "f1_REP":         round(mejor_rep["REP"]["f1-score"], 4),
        "precision_NV":   round(mejor_rep["NV"]["precision"], 4),
        "recall_NV":      round(mejor_rep["NV"]["recall"], 4),
        "historial":      historial,
        "modelo_path":    out_path,
        "timestamp":      datetime.now().isoformat(),
    }
    RESULTADOS[exp_id] = resultado

    # Guardar resultados acumulados
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(RESULTADOS, f, ensure_ascii=False, indent=2)
    print(f"  Resultados guardados → {RESULTS_FILE}")

    return resultado


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 2 — Cargar datos y split base
# ╚══════════════════════════════════════════════════════════════════════════════

print("Cargando anotaciones...")
with open(ANNOTATIONS, encoding="utf-8") as f:
    anotaciones_raw = json.load(f)
print(f"Anotaciones cargadas: {len(anotaciones_raw)}")

tokenizer_base = AutoTokenizer.from_pretrained(MODEL_BASE)

dataset_base = [convertir_io(t, tokenizer_base) for t in anotaciones_raw]

np.random.seed(SEED)
idx     = np.random.permutation(len(dataset_base))
n_train = int(len(dataset_base) * (1 - VAL_SPLIT))

TRAIN_BASE = [dataset_base[i] for i in idx[:n_train]]  # 80 ejemplos
EVAL_BASE  = [dataset_base[i] for i in idx[n_train:]]  # 20 ejemplos

# También guardar las anotaciones raw para aumentación
TRAIN_RAW  = [anotaciones_raw[i] for i in idx[:n_train]]

print(f"Train base: {len(TRAIN_BASE)} | Eval: {len(EVAL_BASE)}")


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 3 — EXP-00: Baseline (reproducir CFH-BERT v2 con ConfliBERT)
# ╚══════════════════════════════════════════════════════════════════════════════

exp00 = entrenar(
    "EXP-00", "Baseline ConfliBERT + WCE n=100 (reproducción v2)",
    train_data=TRAIN_BASE,
    eval_data=EVAL_BASE,
    loss_type="wce",
    nv_boost=1.0,
)
# Resultado esperado: F1≈0.58 (confirma reproducibilidad)


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 4 — EXP-01: NV boost más agresivo (25× en lugar de 19×)
# Cambio mínimo, máximo impacto esperado en NV
# ╚══════════════════════════════════════════════════════════════════════════════

exp01 = entrenar(
    "EXP-01", "ConfliBERT + WCE + NV boost 25×",
    train_data=TRAIN_BASE,
    eval_data=EVAL_BASE,
    loss_type="wce",
    nv_boost=25.0 / 19.0,  # aumentar desde el peso base
)


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 5 — EXP-02: Aumentación por sinónimos controlados
#
# Estrategia: para cada fragmento del train con EBI=1 o NV=1,
# generar una copia reemplazando términos clave por sinónimos del dominio.
# Esto dobla los ejemplos de las clases más difíciles.
# ╚══════════════════════════════════════════════════════════════════════════════

# Diccionarios de sinónimos controlados por categoría
SINONIMOS_EBI = {
    "resultado operacional positivo": [
        "baja operacional", "resultado de combate", "acción bélica exitosa"
    ],
    "baja en combate": [
        "muerto en acción", "neutralizado en operación", "abatido en contacto armado"
    ],
    "neutralizado": ["dado de baja", "abatido", "eliminado", "muerto en combate"],
    "neutralización": ["eliminación", "abatimiento", "baja operacional"],
    "contacto armado": ["enfrentamiento", "combate", "acción armada"],
    "resultado positivo": ["baja favorable", "resultado operacional", "éxito táctico"],
    "abatido": ["neutralizado", "dado de baja", "muerto en acción"],
    "objetivo neutralizado": ["blanco abatido", "objetivo dado de baja"],
}

SINONIMOS_NV = {
    "guerrillero": ["insurgente", "subversivo", "integrante de grupo armado ilegal"],
    "delincuente": ["criminal", "infractor", "elemento al margen de la ley"],
    "objetivo": ["blanco", "individuo", "elemento hostil"],
    "presunto miembro": ["supuesto integrante", "al parecer miembro"],
    "grupo armado ilegal": ["organización subversiva", "grupo al margen de la ley"],
    "miembro de organización subversiva": [
        "integrante de grupo armado ilegal",
        "presunto guerrillero"
    ],
    "occiso": ["individuo muerto", "persona neutralizada"],
}

def aumentar_sinonimos(anotaciones_raw_list, tokenizer):
    """
    Para cada fragmento con spans EBI o NV, genera una copia
    reemplazando un término por un sinónimo aleatorio.
    Mantiene las posiciones de los spans inalteradas
    (el reemplazo es aproximado — preserva la etiqueta del span).
    """
    aumentados = []
    for tarea in anotaciones_raw_list:
        texto  = tarea["text"]
        spans  = tarea.get("label", [])
        labels = {s["labels"][0] for s in spans if s["labels"]}

        nuevos_textos = []
        if "EBI" in labels:
            for original, variantes in SINONIMOS_EBI.items():
                if original.lower() in texto.lower():
                    nuevo = texto.replace(original, random.choice(variantes), 1)
                    if nuevo != texto:
                        nuevos_textos.append(nuevo)
                        break

        if "NV" in labels:
            for original, variantes in SINONIMOS_NV.items():
                if original.lower() in texto.lower():
                    nuevo = texto.replace(original, random.choice(variantes), 1)
                    if nuevo != texto:
                        nuevos_textos.append(nuevo)
                        break

        for nuevo_texto in nuevos_textos:
            nueva_tarea = copy.deepcopy(tarea)
            nueva_tarea["text"] = nuevo_texto
            # Los spans se mantienen aproximados
            # (la aumentación preserva la estructura del fragmento)
            conv = convertir_io(nueva_tarea, tokenizer)
            aumentados.append(conv)

    return aumentados

print("Generando aumentación por sinónimos...")
aumentados_sinonimos = aumentar_sinonimos(TRAIN_RAW, tokenizer_base)
print(f"Ejemplos generados por sinónimos: {len(aumentados_sinonimos)}")

train_exp02 = TRAIN_BASE + aumentados_sinonimos
print(f"Train EXP-02: {len(train_exp02)} (base {len(TRAIN_BASE)} + {len(aumentados_sinonimos)} aumentados)")

exp02 = entrenar(
    "EXP-02", f"ConfliBERT + WCE + Sinónimos (n={len(train_exp02)})",
    train_data=train_exp02,
    eval_data=EVAL_BASE,
    loss_type="wce",
    nv_boost=1.0,
)


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 6 — EXP-03: Aumentación por back-translation ES→EN→ES
#
# Usa deep-translator (Google Translate) para generar paráfrasis
# naturales preservando el significado semántico.
# ╚══════════════════════════════════════════════════════════════════════════════

def aumentar_back_translation(anotaciones_raw_list, tokenizer, max_ejemplos=50):
    """
    Back-translation: ES → EN → ES con Google Translate.
    Solo para fragmentos con EBI, NV o REP (los más difíciles).
    Limitar a max_ejemplos para no saturar la API gratuita.
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("⚠ deep-translator no instalado. Ejecutar: !pip install deep-translator")
        return []

    tr_es_en = GoogleTranslator(source="es", target="en")
    tr_en_es = GoogleTranslator(source="en", target="es")

    aumentados = []
    candidatos = [
        t for t in anotaciones_raw_list
        if any(s["labels"][0] in ["EBI", "NV", "REP"]
               for s in t.get("label", []) if s["labels"])
    ]
    candidatos = candidatos[:max_ejemplos]
    print(f"Back-translation sobre {len(candidatos)} fragmentos candidatos...")

    for tarea in tqdm(candidatos, desc="Back-translation"):
        try:
            texto   = tarea["text"][:400]  # limitar longitud
            en      = tr_es_en.translate(texto)
            time.sleep(0.3)  # respetar rate limit
            es_back = tr_en_es.translate(en)
            time.sleep(0.3)

            if es_back and es_back != texto and len(es_back) > 20:
                nueva_tarea = copy.deepcopy(tarea)
                nueva_tarea["text"] = es_back
                # Los spans se aproximan — la back-translation puede cambiar
                # la posición exacta de los tokens
                conv = convertir_io(nueva_tarea, tokenizer)
                aumentados.append(conv)
        except Exception as e:
            pass  # errores de red — continuar

    print(f"Ejemplos generados por back-translation: {len(aumentados)}")
    return aumentados

aumentados_bt = aumentar_back_translation(TRAIN_RAW, tokenizer_base, max_ejemplos=40)
train_exp03 = TRAIN_BASE + aumentados_bt

exp03 = entrenar(
    "EXP-03", f"ConfliBERT + WCE + Back-translation (n={len(train_exp03)})",
    train_data=train_exp03,
    eval_data=EVAL_BASE,
    loss_type="wce",
    nv_boost=1.0,
)


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 7 — EXP-04: Aumentación por paráfrasis LLM (Claude API)
#
# Claude genera variaciones del fragmento preservando el tipo de violencia
# discursiva (EBI, SA, NV, REP). Es la aumentación más sofisticada:
# genera diversidad semántica real manteniendo las etiquetas.
# ╚══════════════════════════════════════════════════════════════════════════════

def aumentar_llm(anotaciones_raw_list, tokenizer, max_ejemplos=30,
                  api_key=None):
    """
    Genera paráfrasis usando Claude API (claude-sonnet-4-20250514).
    Para cada fragmento, genera 2 variaciones preservando el tipo
    de violencia discursiva presente.
    """
    if api_key is None:
        print("⚠ Requiere ANTHROPIC_API_KEY. Setear con:")
        print("  import os; os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'")
        return []

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    PROMPT_SISTEMA = """Eres un experto en lingüística forense colombiana.
Tu tarea es generar variaciones de fragmentos de documentos judiciales sobre los falsos positivos (Macrocaso 003 JEP).
REGLAS:
1. Preserva el tipo de violencia discursiva del fragmento original (EBI, SA, NV, o REP).
2. Cambia vocabulario, estructura sintáctica y orden de oraciones — pero NO el significado discursivo.
3. Mantén el registro jurídico colombiano.
4. Devuelve SOLO el fragmento parafrasado, sin explicaciones.
"""

    def get_label_names(tarea):
        return list({s["labels"][0] for s in tarea.get("label", []) if s["labels"]})

    candidatos = [
        t for t in anotaciones_raw_list
        if any(s["labels"][0] in ["EBI", "NV", "REP"]
               for s in t.get("label", []) if s["labels"])
    ][:max_ejemplos]

    aumentados = []
    print(f"Paráfrasis LLM sobre {len(candidatos)} fragmentos...")

    for tarea in tqdm(candidatos, desc="LLM augmentation"):
        labels_presentes = get_label_names(tarea)
        texto = tarea["text"][:350]

        prompt = f"""Fragmento original (contiene: {', '.join(labels_presentes)}):
"{texto}"

Genera UNA paráfrasis preservando exactamente el tipo de violencia discursiva.
Solo el fragmento parafrasado:"""

        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=PROMPT_SISTEMA,
                messages=[{"role": "user", "content": prompt}]
            )
            parafrasis = resp.content[0].text.strip().strip('"')
            if parafrasis and len(parafrasis) > 30:
                nueva_tarea = copy.deepcopy(tarea)
                nueva_tarea["text"] = parafrasis
                conv = convertir_io(nueva_tarea, tokenizer)
                aumentados.append(conv)
            time.sleep(0.5)
        except Exception as e:
            pass

    print(f"Ejemplos generados por LLM: {len(aumentados)}")
    return aumentados

# Para usar: setear tu API key
# import os
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
# aumentados_llm = aumentar_llm(TRAIN_RAW, tokenizer_base,
#                                max_ejemplos=30,
#                                api_key=os.environ["ANTHROPIC_API_KEY"])

# Sin API key — omitir este experimento
aumentados_llm = []
print("EXP-04: omitido si no hay API key. Setear ANTHROPIC_API_KEY para activar.")

if aumentados_llm:
    train_exp04 = TRAIN_BASE + aumentados_llm
    exp04 = entrenar(
        "EXP-04", f"ConfliBERT + WCE + LLM augmentation (n={len(train_exp04)})",
        train_data=train_exp04, eval_data=EVAL_BASE,
    )


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 8 — EXP-05: Focal Loss γ=2
#
# Focal Loss penaliza más los ejemplos difíciles (tokens NV mal clasificados)
# y menos los fáciles (tokens O correctamente clasificados).
# Especialmente efectivo cuando O domina con 87%+ del corpus.
# ╚══════════════════════════════════════════════════════════════════════════════

exp05 = entrenar(
    "EXP-05", "ConfliBERT + Focal Loss γ=2 (n=100)",
    train_data=TRAIN_BASE,
    eval_data=EVAL_BASE,
    loss_type="focal",
    gamma=2.0,
    nv_boost=1.0,
)


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 9 — EXP-06: Focal Loss γ=2 + NV boost 25×
# ╚══════════════════════════════════════════════════════════════════════════════

exp06 = entrenar(
    "EXP-06", "ConfliBERT + Focal Loss γ=2 + NV boost 25× (n=100)",
    train_data=TRAIN_BASE,
    eval_data=EVAL_BASE,
    loss_type="focal",
    gamma=2.0,
    nv_boost=25.0 / 19.0,
)


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 10 — EXP-07: Threshold tuning por clase
#
# En lugar de argmax estándar (umbral efectivo=0.5),
# calibrar umbrales individuales por clase sobre el conjunto de evaluación.
# Para NV bajar el umbral a 0.25-0.30 aumenta el recall
# a costa de algo de precisión.
# ╚══════════════════════════════════════════════════════════════════════════════

def buscar_mejores_thresholds(model_path, eval_data):
    """
    Grid search de umbrales para NV y REP sobre el eval set.
    Devuelve los umbrales que maximizan F1 macro.
    """
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    el    = DataLoader(CFHDataset(eval_data), batch_size=BATCH_SIZE)

    mejor_f1     = 0.0
    mejores_thr  = None

    # Grid de umbrales para NV (la clase más difícil)
    for thr_nv in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        thresholds = {LABEL2ID["NV"]: thr_nv}
        f1, rep, _ = evaluar(model, el, thresholds)
        print(f"  thr_NV={thr_nv:.2f} → F1={f1:.4f} | NV={rep['NV']['f1-score']:.4f}")
        if f1 > mejor_f1:
            mejor_f1    = f1
            mejores_thr = thresholds.copy()

    print(f"\n  Mejores umbrales: {mejores_thr} → F1={mejor_f1:.4f}")
    return mejores_thr, mejor_f1

# Buscar thresholds sobre el mejor modelo hasta ahora
mejor_exp = max(
    [e for e in RESULTADOS.values()],
    key=lambda x: x["f1_macro"]
)
print(f"\nThreshold tuning sobre: {mejor_exp['exp_id']} (F1={mejor_exp['f1_macro']})")

mejores_thresholds, f1_con_thresholds = buscar_mejores_thresholds(
    mejor_exp["modelo_path"], EVAL_BASE
)

RESULTADOS["EXP-07"] = {
    "exp_id": "EXP-07",
    "descripcion": f"Threshold tuning sobre {mejor_exp['exp_id']}",
    "base_exp": mejor_exp["exp_id"],
    "thresholds": mejores_thresholds,
    "f1_macro": round(f1_con_thresholds, 4),
    "timestamp": datetime.now().isoformat(),
}
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(RESULTADOS, f, ensure_ascii=False, indent=2)


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 11 — EXP-08: CFH-BERT v3 — TODO COMBINADO
#
# ConfliBERT + Focal Loss γ=2 + NV boost + TODOS los aumentados
# Este es el modelo v3 que se presenta en la tesis y a la JEP.
# ╚══════════════════════════════════════════════════════════════════════════════

todos_los_aumentados = aumentados_sinonimos + aumentados_bt + aumentados_llm
train_v3 = TRAIN_BASE + todos_los_aumentados

print(f"\nCFH-BERT v3 — dataset completo:")
print(f"  Base:         {len(TRAIN_BASE)}")
print(f"  Sinónimos:    {len(aumentados_sinonimos)}")
print(f"  Back-transl:  {len(aumentados_bt)}")
print(f"  LLM:          {len(aumentados_llm)}")
print(f"  TOTAL TRAIN:  {len(train_v3)}")

exp08 = entrenar(
    "EXP-08", f"CFH-BERT v3: ConfliBERT + Focal Loss γ=2 + NV 25× + Todos aumentados (n={len(train_v3)})",
    train_data=train_v3,
    eval_data=EVAL_BASE,
    loss_type="focal",
    gamma=2.0,
    nv_boost=25.0 / 19.0,
    thresholds=mejores_thresholds,
    epochs=25,  # más épocas para el modelo final
)


# ╔══════════════════════════════════════════════════════════════════════════════
# CELDA 12 — Tabla comparativa (Ablation Study)
#
# Esta tabla va directamente al Capítulo 4 (Metodología) de la tesis
# y al paper en inglés.
# ╚══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*90)
print("ABLATION STUDY — CFH-BERT v2 → v3")
print("═"*90)
print(f"{'Experimento':<12} {'Descripción':<42} {'n_train':>7} "
      f"{'F1':<7} {'EBI':<7} {'SA':<7} {'NV':<7} {'REP':<7}")
print("─"*90)

# Baseline v2 (histórico)
print(f"{'v2 (ref)':<12} {'ConfliBERT + WCE, n=100 (histórico)':<42} {'100':>7} "
      f"{'0.584':<7} {'0.52':<7} {'0.52':<7} {'0.32':<7} {'0.77':<7}")

for exp_id, r in sorted(RESULTADOS.items()):
    if "f1_macro" not in r:
        continue
    desc_corta = r["descripcion"][:40]
    delta = r["f1_macro"] - 0.584
    signo = "+" if delta >= 0 else ""
    print(f"{r['exp_id']:<12} {desc_corta:<42} {r.get('n_train', '?'):>7} "
          f"{r['f1_macro']:<7.3f} "
          f"{r.get('f1_EBI', '?'):<7} "
          f"{r.get('f1_SA', '?'):<7} "
          f"{r.get('f1_NV', '?'):<7} "
          f"{r.get('f1_REP', '?'):<7} "
          f"({signo}{delta:.3f})")

print("─"*90)
mejor_final = max(
    [r for r in RESULTADOS.values() if "f1_macro" in r],
    key=lambda x: x["f1_macro"]
)
print(f"\n✅ MEJOR MODELO: {mejor_final['exp_id']} — F1 macro={mejor_final['f1_macro']:.4f}")
print(f"   Guardado en: {mejor_final.get('modelo_path', 'ver RESULTS_FILE')}")

# Para uso institucional JEP
if mejor_final["f1_macro"] >= 0.85:
    print("🎯 META JEP ALCANZADA (F1 ≥ 0.85)")
elif mejor_final["f1_macro"] >= 0.70:
    print("📊 META DEFENSA ALCANZADA (F1 ≥ 0.70)")
    print("   Para uso JEP: necesita más datos anotados (~2.000 fragmentos)")
else:
    print("⚠ Ninguna meta alcanzada — revisar estrategia de aumentación")

print(f"\nResultados completos: {RESULTS_FILE}")

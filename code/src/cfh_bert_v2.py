"""
CFH-BERT v2 — Fine-tuning con IO tagging
=========================================
Mejoras respecto a v1:
- Esquema IO (5 clases) en lugar de BIO (9 clases)
- Doble soporte por clase al eliminar etiquetas B-
- Weighted loss para clases desbalanceadas
- Evaluación con F1 macro por clase (no solo accuracy)
- Guardado del mejor modelo según F1 macro eval

Uso:
    python cfh_bert_v2.py --annotations annotations_mireya_v1.json.json
                          --epochs 15
                          --output cfh-bert-v2

Requiere:
    pip install transformers torch sklearn tqdm
"""

import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import os

# ── Configuración ─────────────────────────────────────────────────────────────
MODEL_NAME  = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH  = 512
BATCH_SIZE  = 8
LEARNING_RATE = 2e-5

# Esquema IO — 5 clases
LABEL2ID = {"O": 0, "EBI": 1, "SA": 2, "NV": 3, "REP": 4}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# ── Conversión de anotaciones ──────────────────────────────────────────────────
def convertir_io(tarea, tokenizer):
    """Convierte una tarea Label Studio a tokens + etiquetas IO."""
    texto = tarea["text"]
    spans = tarea.get("label", [])

    # Etiqueta por carácter
    char_labels = ["O"] * len(texto)
    for span in spans:
        label = span["labels"][0] if span["labels"] else "O"
        if label == "O":
            continue
        for i in range(span["start"], min(span["end"], len(texto))):
            char_labels[i] = label  # IO: misma etiqueta para todo el span

    # Tokenizar con offsets
    encoding = tokenizer(
        texto,
        max_length=MAX_LENGTH,
        truncation=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    token_labels = []
    for start, end in encoding["offset_mapping"]:
        if start == end:
            token_labels.append(-100)  # token especial
        else:
            label = char_labels[start]
            token_labels.append(LABEL2ID.get(label, 0))

    return {
        "input_ids":      encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels":         token_labels,
    }


# ── Dataset ───────────────────────────────────────────────────────────────────
class CFHDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.data[idx].items()}


# ── Pesos de clase para loss balanceada ───────────────────────────────────────
def calcular_pesos(dataset_list):
    """Calcula pesos inversos a la frecuencia de cada clase."""
    conteos = [0] * len(LABEL2ID)
    for d in dataset_list:
        for label in d["labels"]:
            if label >= 0:
                conteos[label] += 1
    total = sum(conteos)
    pesos = [total / (len(LABEL2ID) * max(c, 1)) for c in conteos]
    print("Distribución de clases:")
    for lid, (count, peso) in enumerate(zip(conteos, pesos)):
        print(f"  {ID2LABEL[lid]:6s}: {count:5d} tokens  peso={peso:.2f}")
    return torch.tensor(pesos, dtype=torch.float)


# ── Evaluación ────────────────────────────────────────────────────────────────
def evaluar(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(-1)
            mask  = batch["labels"] != -100
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(batch["labels"][mask].cpu().numpy())

    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report   = classification_report(
        all_labels, all_preds,
        target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))],
        zero_division=0
    )
    return f1_macro, report


# ── Entrenamiento principal ───────────────────────────────────────────────────
def entrenar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Cargar anotaciones
    with open(args.annotations, encoding="utf-8") as f:
        anotaciones = json.load(f)
    print(f"Anotaciones cargadas: {len(anotaciones)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Convertir a IO
    dataset = [convertir_io(t, tokenizer) for t in anotaciones]

    # Split 80/20 estratificado por corpus_type
    np.random.seed(42)
    idx = np.random.permutation(len(dataset))
    n_train = int(len(dataset) * 0.8)
    train_data = [dataset[i] for i in idx[:n_train]]
    eval_data  = [dataset[i] for i in idx[n_train:]]
    print(f"Train: {len(train_data)} | Eval: {len(eval_data)}")

    # Pesos de clase
    pesos = calcular_pesos(train_data).to(device)

    # DataLoaders
    train_loader = DataLoader(CFHDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    eval_loader  = DataLoader(CFHDataset(eval_data),  batch_size=BATCH_SIZE)

    # Modelo
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Loss con pesos
    loss_fn = torch.nn.CrossEntropyLoss(weight=pesos, ignore_index=-100)

    # Entrenamiento
    mejor_f1   = 0.0
    mejor_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Época {epoch+1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            # Loss con pesos manualmente
            logits = outputs.logits  # (B, T, C)
            loss = loss_fn(
                logits.view(-1, len(LABEL2ID)),
                batch["labels"].view(-1)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # Evaluación
        f1_macro, report = evaluar(model, eval_loader, device)
        print(f"\nÉpoca {epoch+1} — Loss: {total_loss/len(train_loader):.3f} | F1 macro: {f1_macro:.3f}")

        if f1_macro > mejor_f1:
            mejor_f1    = f1_macro
            mejor_epoch = epoch + 1
            # Guardar mejor modelo
            os.makedirs(args.output, exist_ok=True)
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"  ✓ Mejor modelo guardado (época {mejor_epoch}, F1={mejor_f1:.3f})")

    # Reporte final
    print(f"\n{'='*50}")
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Mejor F1 macro: {mejor_f1:.3f} (época {mejor_epoch})")
    print(f"Modelo guardado en: {args.output}/")
    print(f"\nReporte final (mejor modelo):")

    # Cargar mejor modelo y evaluar
    best_model = AutoModelForTokenClassification.from_pretrained(args.output).to(device)
    _, report = evaluar(best_model, eval_loader, device)
    print(report)

    # Guardar reporte
    with open(f"{args.output}/reporte_eval.txt", "w", encoding="utf-8") as f:
        f.write(f"CFH-BERT v2 — Reporte de evaluación\n")
        f.write(f"Mejor época: {mejor_epoch} | F1 macro: {mejor_f1:.3f}\n\n")
        f.write(report)
    print(f"Reporte guardado en {args.output}/reporte_eval.txt")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFH-BERT v2 — IO tagging")
    parser.add_argument("--annotations", default="annotations_mireya_v1.json.json",
                        help="Archivo JSON de anotaciones Label Studio")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Número de épocas de entrenamiento")
    parser.add_argument("--output", default="cfh-bert-v2",
                        help="Directorio de salida del modelo")
    args = parser.parse_args()
    entrenar(args)

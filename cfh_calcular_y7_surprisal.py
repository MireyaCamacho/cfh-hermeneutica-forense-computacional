# -*- coding: utf-8 -*-
"""
cfh_calcular_y7_surprisal.py
================================================================================
Calcula y7 (surprisal contrastivo) para Corpus A y B, y lo escribe en los
CSV de indicadores del SEM. Desbloquea la estimacion del modelo latente completo.

DEFINICION
----------
y7 = PLL_BETO - PLL_ConfliBERT   (por seccion)

  - PLL = pseudo-log-likelihood promedio por token (masked language modeling).
  - PLL_BETO       : que tan "esperable" es el texto para un modelo de espanol general.
  - PLL_ConfliBERT : que tan "esperable" es para un modelo de conflicto/violencia politica.
  - y7 > 0  => el texto es MAS predecible para ConfliBERT que para BETO
              (lenguaje mas cercano al registro belico-institucional del conflicto).
  - y7 < 0  => el texto es mas "civil"/general que belico.

  Nota interpretativa: se reporta la DIFERENCIA de log-likelihood. Como PLL es
  log-verosimilitud (mayor = mas predecible), un texto tipico del registro del
  conflicto tendra PLL_ConfliBERT alto => y7 negativo o cercano a 0 hacia lo belico.
  El signo se documenta en la tesis segun la orientacion final que decida Mireya;
  aqui se deja el valor crudo y reproducible.

FUENTE DE DATOS
---------------
  data/processed/corpus_a/{hash16}.json  + {hash16}.txt   (texto limpio)
  data/processed/corpus_b/*.json         + *.txt
  data/processed/corpus_b_json/*.json    + *.txt          (por si B esta aqui)

  - El doc_id del SEM == sha256_clean del JSON == nombre del .txt gemelo.
  - Cada seccion se corta del .txt usando segmentation.sections[i].char_range.
  - Solo se procesan las secciones is_target=True (las que usa el SEM).

SALIDA
------
  Escribe la columna y7_surprisal en:
    data/features/indicators_corpus_a.csv   (819 filas)
    data/features/indicators_corpus_b.csv   (54 filas)
  Hace BACKUP con timestamp antes de sobrescribir.

OPTIMIZACION (CPU)
------------------
  - Muestreo de tokens: enmascara 1 de cada STRIDE tokens (default 4).
  - Trunca cada seccion a MAX_TOKENS (default 512) para acotar el costo.
  - Con esto el calculo baja de horas a minutos en CPU. En GPU es casi instantaneo.

USO
---
  conda activate cfh
  python cfh_calcular_y7_surprisal.py                 # corre todo (A y B)
  python cfh_calcular_y7_surprisal.py --dry-run       # procesa 5 docs, no escribe
  python cfh_calcular_y7_surprisal.py --stride 2      # mas preciso, mas lento
  python cfh_calcular_y7_surprisal.py --device cuda   # si hay GPU (Colab)

REQUISITOS
----------
  transformers, torch, pandas, numpy  (ya presentes en el env cfh)
================================================================================
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ------------------------------------------------------------------ CONFIG ----
ROOT = Path(__file__).resolve().parent
# Permitir que el script viva en cualquier carpeta pero apunte al repo:
# si no encuentra data/, sube directorios buscandola.
def _find_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "data" / "features" / "indicators_corpus_a.csv").exists():
            return cand
    return start

REPO = _find_repo_root(ROOT)

PROCESSED_A = REPO / "data" / "processed" / "corpus_a"
# Corpus B: SOLO corpus_b_json (JSON con sha256_clean + segmentation.char_range,
# alineados con el doc_id del SEM). La carpeta corpus_b/ tiene JSON viejos con
# estructura distinta (doc_id=radicado, seccion=CUERPO) que NO coinciden con el SEM.
PROCESSED_B_DIRS = [
    REPO / "data" / "processed" / "corpus_b_json",
]

CSV_A = REPO / "data" / "features" / "indicators_corpus_a.csv"
CSV_B = REPO / "data" / "features" / "indicators_corpus_b.csv"

MODEL_CONFLIBERT = "eventdata-utd/ConfliBERT-Spanish-BETO-Cased-v1"
MODEL_BETO = "dccuchile/bert-base-spanish-wwm-cased"

MAX_TOKENS = 512     # truncado por seccion (limite BERT)
STRIDE = 4           # enmascarar 1 de cada STRIDE tokens (muestreo)
MIN_TOKENS = 5       # secciones mas cortas que esto -> y7 = NaN (no fiable)
MAX_MASK_POS = 48    # tope de posiciones enmascaradas por seccion (cota el costo)
BATCH_POS = 16       # posiciones procesadas por forward-pass (batching)

# --------------------------------------------------------------- MODELO PLL ---
class MaskedLMScorer:
    """Calcula pseudo-log-likelihood promedio por token via masked LM."""

    def __init__(self, model_name: str, device: str):
        self.name = model_name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        self.mask_id = self.tok.mask_token_id

    @torch.no_grad()
    def pll(self, text: str, max_tokens: int = MAX_TOKENS, stride: int = STRIDE) -> float:
        """
        Pseudo-log-likelihood promedio por token enmascarado.
        Enmascara 1 de cada `stride` tokens (excluyendo [CLS]/[SEP]) y promedia
        el log-prob del token verdadero en cada posicion enmascarada.
        Retorna NaN si el texto es demasiado corto.

        OPTIMIZADO: en vez de un forward-pass por posicion (lento en CPU), agrupa
        las posiciones enmascaradas en BATCHES y las procesa juntas. Cada fila del
        batch es una copia de la secuencia con UNA posicion enmascarada. Esto reduce
        ~50x el numero de llamadas al modelo.
        """
        # Cortar el texto ANTES de tokenizar para no procesar 60k+ chars de golpe.
        # ~6 chars/token * max_tokens da margen de sobra; evita picos de RAM/CPU.
        if len(text) > max_tokens * 8:
            text = text[: max_tokens * 8]

        enc = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )
        input_ids = enc["input_ids"][0]
        n = input_ids.size(0)
        # posiciones enmascarables: excluir primer ([CLS]) y ultimo ([SEP])
        maskable = list(range(1, n - 1))
        if len(maskable) < MIN_TOKENS:
            return float("nan")
        positions = maskable[::stride]
        # Tope duro de posiciones por seccion (cota el costo de secciones largas).
        if len(positions) > MAX_MASK_POS:
            # muestreo uniforme de MAX_MASK_POS posiciones
            idx = np.linspace(0, len(positions) - 1, MAX_MASK_POS).astype(int)
            positions = [positions[i] for i in idx]
        if not positions:
            return float("nan")

        true_ids = [int(input_ids[p].item()) for p in positions]

        # Construir batch: (num_pos, seq_len), cada fila con una posicion enmascarada
        log_probs = []
        for start in range(0, len(positions), BATCH_POS):
            chunk_pos = positions[start:start + BATCH_POS]
            chunk_true = true_ids[start:start + BATCH_POS]
            batch = input_ids.unsqueeze(0).repeat(len(chunk_pos), 1).clone()
            for i, p in enumerate(chunk_pos):
                batch[i, p] = self.mask_id
            batch = batch.to(self.device)
            logits = self.model(batch).logits            # (num_pos, seq_len, vocab)
            for i, (p, tid) in enumerate(zip(chunk_pos, chunk_true)):
                lp = torch.log_softmax(logits[i, p], dim=-1)[tid]
                log_probs.append(float(lp.item()))

        return float(np.mean(log_probs)) if log_probs else float("nan")


# ------------------------------------------------------- LECTURA DE SECCIONES --
def iter_documents(processed_dir: Path):
    """
    Genera (doc_id, section_id, texto_seccion) para cada seccion is_target=True
    de cada JSON en processed_dir que tenga su .txt gemelo.
    """
    if not processed_dir.exists():
        return
    for json_path in sorted(processed_dir.glob("*.json")):
        if json_path.name.startswith("batch_summary"):
            continue
        txt_path = json_path.with_suffix(".txt")
        if not txt_path.exists():
            # algunos B tienen sufijo _sentencia; intentar variantes
            alt = list(processed_dir.glob(json_path.stem + "*.txt"))
            if not alt:
                print(f"  [WARN] sin .txt para {json_path.name}, se omite")
                continue
            txt_path = alt[0]

        try:
            d = json.load(open(json_path, encoding="utf-8"))
        except Exception as e:
            print(f"  [WARN] no se pudo leer {json_path.name}: {e}")
            continue

        doc_id = d.get("sha256_clean") or d.get("metadata", {}).get("doc_id")
        if not doc_id:
            print(f"  [WARN] sin doc_id en {json_path.name}, se omite")
            continue

        clean_text = txt_path.read_text(encoding="utf-8", errors="replace")
        sections = d.get("segmentation", {}).get("sections", [])
        for sec in sections:
            if not sec.get("is_target", False):
                continue
            sid = sec.get("section_id")
            rng = sec.get("char_range")
            if not sid or not rng or len(rng) != 2:
                continue
            a, b = rng
            seg = clean_text[a:b].strip()
            if not seg:
                continue
            yield doc_id, sid, seg


# --------------------------------------------------------------------- MAIN ---
def procesar_corpus(nombre, processed_dirs, csv_path, scorers, dry_run, stride, limit=None):
    print(f"\n{'='*70}\nCORPUS {nombre}\n{'='*70}")
    print(f"CSV objetivo: {csv_path}")
    if not csv_path.exists():
        print(f"  [ERROR] no existe {csv_path}, se omite corpus {nombre}")
        return None

    df = pd.read_csv(csv_path)
    print(f"  filas en CSV: {len(df)}")
    if "y7_surprisal" not in df.columns:
        df["y7_surprisal"] = np.nan

    conflibert, beto = scorers
    resultados = {}   # (doc_id, section_id) -> y7
    procesados = 0

    # --- CHECKPOINT: retomar progreso previo si existe ---
    ckpt_path = csv_path.with_name(f"_ckpt_y7_{nombre}.json")
    if ckpt_path.exists() and not dry_run:
        try:
            prev = json.load(open(ckpt_path, encoding="utf-8"))
            # las claves se guardan como "doc_id||section_id"
            for k, v in prev.items():
                d, s = k.split("||", 1)
                resultados[(d, s)] = v
            print(f"  [checkpoint] retomando {len(resultados)} secciones ya calculadas")
        except Exception as e:
            print(f"  [checkpoint] no se pudo leer ({e}), empiezo de cero")

    def _guardar_ckpt():
        if dry_run:
            return
        serial = {f"{d}||{s}": v for (d, s), v in resultados.items()}
        tmp = ckpt_path.with_suffix(".tmp")
        json.dump(serial, open(tmp, "w", encoding="utf-8"))
        tmp.replace(ckpt_path)   # escritura atomica

    for pdir in processed_dirs:
        for doc_id, sid, texto in iter_documents(pdir):
            key = (doc_id, sid)
            if key in resultados:
                continue   # ya calculado (checkpoint) -> saltar
            pll_conf = conflibert.pll(texto, stride=stride)
            pll_beto = beto.pll(texto, stride=stride)
            if np.isnan(pll_conf) or np.isnan(pll_beto):
                y7 = float("nan")
            else:
                y7 = pll_beto - pll_conf
            resultados[key] = y7
            procesados += 1
            if procesados <= 10 or procesados % 10 == 0:
                print(f"  [{procesados}] {doc_id[:12]} / {sid:20s} "
                      f"PLL_beto={pll_beto:7.3f} PLL_conf={pll_conf:7.3f} y7={y7:7.3f}",
                      flush=True)
            # guardar checkpoint cada 50 secciones
            if procesados % 50 == 0:
                _guardar_ckpt()
            if limit and procesados >= limit:
                print(f"  [dry-run] alcanzado limite {limit}, corto.")
                break
        if limit and procesados >= limit:
            break

    _guardar_ckpt()   # checkpoint final antes de escribir el CSV
    print(f"\n  secciones procesadas (nuevas esta corrida): {procesados}")
    print(f"  claves unicas totales (doc_id, section_id): {len(resultados)}")

    # --- alinear con el CSV por (doc_id, section_id) ---
    def _lookup(row):
        return resultados.get((row["doc_id"], row["section_id"]), np.nan)

    df["_y7_nuevo"] = df.apply(_lookup, axis=1)
    n_match = df["_y7_nuevo"].notna().sum()
    n_total = len(df)
    print(f"  filas del CSV emparejadas: {n_match} / {n_total}")

    faltantes = df[df["_y7_nuevo"].isna()][["doc_id", "section_id"]]
    if len(faltantes):
        print(f"  [AVISO] {len(faltantes)} filas sin y7 (texto corto o sin JSON):")
        for _, r in faltantes.head(15).iterrows():
            print(f"      {r['doc_id'][:16]} / {r['section_id']}")
        if len(faltantes) > 15:
            print(f"      ... y {len(faltantes)-15} mas")

    if dry_run:
        print("  [dry-run] NO se escribe el CSV.")
        return df

    # --- backup + escritura ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = csv_path.with_name(csv_path.stem + f"_BACKUP_pre_y7_{ts}.csv")
    shutil.copy2(csv_path, backup)
    print(f"  backup -> {backup.name}")

    df["y7_surprisal"] = df["_y7_nuevo"]
    df = df.drop(columns=["_y7_nuevo"])
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  ESCRITO: {csv_path.name}  (y7_surprisal actualizado)")
    # exito -> borrar checkpoint para dejar limpio
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"  checkpoint eliminado ({ckpt_path.name})")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="procesa pocos docs y no escribe los CSV")
    ap.add_argument("--stride", type=int, default=STRIDE,
                    help=f"enmascarar 1 de cada N tokens (default {STRIDE}; menor=mas preciso/lento)")
    ap.add_argument("--device", default=None,
                    help="cpu | cuda  (default: auto)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    limit = 5 if args.dry_run else None

    print("="*70)
    print("CFH y7 surprisal contrastivo  (ConfliBERT vs BETO)")
    print("="*70)
    print(f"REPO:   {REPO}")
    print(f"device: {device}   stride: {args.stride}   max_tokens: {MAX_TOKENS}")
    print(f"dry-run: {args.dry_run}")

    print("\nCargando modelos (puede tardar la primera vez)...")
    conflibert = MaskedLMScorer(MODEL_CONFLIBERT, device)
    print(f"  OK ConfliBERT ({MODEL_CONFLIBERT})")
    beto = MaskedLMScorer(MODEL_BETO, device)
    print(f"  OK BETO ({MODEL_BETO})")
    scorers = (conflibert, beto)

    df_a = procesar_corpus("A", [PROCESSED_A], CSV_A, scorers,
                           args.dry_run, args.stride, limit)
    df_b = procesar_corpus("B", PROCESSED_B_DIRS, CSV_B, scorers,
                           args.dry_run, args.stride, limit)

    # ------------------------------------------------- ASSERTS DE VALIDACION --
    if not args.dry_run:
        print(f"\n{'='*70}\nVALIDACION FINAL\n{'='*70}")
        ok = True
        for nombre, df, esperado in [("A", df_a, 819), ("B", df_b, 54)]:
            if df is None:
                print(f"  [FALLA] corpus {nombre} no se proceso")
                ok = False
                continue
            n_nan = df["y7_surprisal"].isna().sum()
            n_ok = df["y7_surprisal"].notna().sum()
            rng = (df["y7_surprisal"].min(), df["y7_surprisal"].max())
            print(f"  Corpus {nombre}: filas={len(df)} (esperado {esperado}) | "
                  f"y7 no-NaN={n_ok} | NaN={n_nan} | rango=[{rng[0]:.3f}, {rng[1]:.3f}]")
            if len(df) != esperado:
                print(f"    [AVISO] filas != esperado ({esperado})")
            if n_nan > 0:
                print(f"    [AVISO] {n_nan} filas con y7=NaN (revisar seccion 'faltantes' arriba)")
        print("\n  Nota: si quedan NaN, suelen ser secciones muy cortas o sin JSON.")
        print("  Para el SEM, semopy puede manejar NaN por listwise deletion, o")
        print("  se imputan con la media de la seccion. Decidir con Mireya.")
        print(f"\n{'OK' if ok else 'CON AVISOS'} — siguiente paso: python run_sem.py")


if __name__ == "__main__":
    main()

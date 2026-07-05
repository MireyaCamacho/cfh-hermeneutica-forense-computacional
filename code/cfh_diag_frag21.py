# -*- coding: utf-8 -*-
"""
CFH — Micro-diagnostico del fragmento #21: clasificar enemigo/guerrilleros
Imprime la oracion completa (sin truncar) y el arbol de dependencias de los
terminos NV, para decidir si es Tipo 3 (NV real contra la victima) o narracion.
"""
import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "code" / "src"
sys.path.insert(0, str(SRC))
GOLD = REPO / "data" / "referencias" / "annotations_mireya_v1.json"

with open(GOLD, encoding="utf-8") as f:
    gold = {d["id"]: d for d in json.load(f)}

frag = gold[21]
texto = frag["text"]

print("=" * 70)
print("FRAGMENTO #21 — texto completo")
print("=" * 70)
print(texto)
print()

import spacy
nlp = spacy.load("es_core_news_lg")
nlp.max_length = 3_000_000
doc = nlp(texto)

print("=" * 70)
print("Oraciones que contienen 'enemigo' o 'guerrill*'")
print("=" * 70)
for sent in doc.sents:
    low = sent.text.lower()
    if "enemigo" in low or "guerrill" in low:
        print(f"\nORACION: {sent.text.strip()}")
        for tok in sent:
            if tok.lemma_.lower() in {"enemigo", "guerrillero"} or "guerrill" in tok.text.lower():
                print(f"  token='{tok.text}' lemma='{tok.lemma_}' pos={tok.pos_} "
                      f"dep={tok.dep_} head='{tok.head.text}' head_pos={tok.head.pos_}")
                # cadena de heads hasta la raiz
                cadena, t = [], tok
                for _ in range(6):
                    cadena.append(f"{t.text}({t.dep_})")
                    if t.head == t:
                        break
                    t = t.head
                print(f"     cadena->raiz: {' -> '.join(cadena)}")

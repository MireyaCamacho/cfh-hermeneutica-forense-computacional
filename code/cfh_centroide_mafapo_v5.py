"""
CFH — Cálculo del centroide MAFAPO v5
======================================
Combina los 25 textos base MAFAPO (informe 2021) con los 194 candidatos
de voz de víctimas extraídos de 7 audiencias de observaciones (v5).

Método idéntico al v4:
  - Embedding CLS: last_hidden_state[:, 0, :]
  - Ponderación: w=1.8 voz directa en audiencias (nivel 1-2), w=1.0 texto escrito base
  - Promedio ponderado, np.save

El centroide CIDH NO cambia (sigue siendo v3 — solo se recalcula MAFAPO).

Ejecutar (CPU local, no requiere GPU):
  conda activate cfh
  python code/cfh_centroide_mafapo_v5.py
"""
import json
import numpy as np
from pathlib import Path

REPO       = Path(__file__).resolve().parent.parent
REF_DIR    = REPO / "data" / "referencias"
CANDIDATOS = REF_DIR / "corpus_c_victimas_v5.json"

# ── 25 textos base MAFAPO (informe "Unidas por la Memoria y la Verdad" 2021) ──
TEXTOS_MAFAPO_BASE = [
    "Con la mano en el alma, pido justicia. Que nos entreguen a la persona que realmente le segó la vida a mi hijo. Para mí es muy triste que en 13 años nadie haya respondido por la muerte de mi hijo.",
    "Recolecta los testimonios, a profundidad, de las madres que llevamos en esta lucha más de trece años, en busca de verdad, justicia y garantías de no repetición. Este informe se elaboró a partir de las entrevistas y encuentros que hemos tenido las víctimas en los últimos años.",
    "Tuve que vender mi casa para autofinanciar la búsqueda de mi hijo. Hasta el momento nadie sabe con certeza qué pasó con el joven, y ella tuvo que mudarse a una casa a las afueras de Bogotá donde vive en condiciones precarias.",
    "Es difícil pensar en acciones restaurativas cuando los principales responsables no han dicho la verdad y no han pedido perdón. El Ejército ha incumplido su promesa.",
    "No respetaron a las víctimas. El general tiene que aceptar esa verdad, que eso lo hizo él.",
    "Nuestros hijos no son una DUDA, la certeza de sus vidas no se relativiza, los parimos y el Estado los asesinó sin lugar a DUDAS.",
    "Si no fueran 6.402, así sean 50, sea uno, fueron crímenes de Estado y no se pueden quedar en el olvido.",
    "Sentimos como si hubieran tirado a nuestros hijos a una fosa común. Aún no nos reponemos. Nosotras seguimos sin encontrar justicia.",
    "Las investigaciones judiciales comprobaron lo que las mujeres de MAFAPO afirmaban: el Ejército de Colombia asesinó a personas inocentes y las presentó como bajas en combate.",
    "Los cuerpos de nuestros familiares, encontrados como falsas bajas en combate en Ocaña, fueron alicorados y drogados antes de ser asesinados. Los trataron peor que a los animales.",
    "Son 6.402 madres y familias que hoy caminan con el corazón despierto buscando la verdad y la justicia, mujeres que le enseñan a este país lo que significa tatuarse el dolor y el amor en el cuerpo.",
    "Los jóvenes de Soacha eran inocentes, trabajadores. El Ejército los reclutó con engaños prometiéndoles trabajo. Los llevaron a lugares que ellos no conocían y los ejecutaron.",
    "Mi hijo salió a buscar trabajo y nunca volvió. ¿Cómo podían volverse guerrilleros y enfrentarse en un combate contra el Ejército en tan pocas horas?",
    "Nos sumamos a esa intención de que en El Copey donde se cree que está el cuerpo del hijo de Doris no se construya ningún pavimento. Solicitamos comedidamente que esos cuerpos no queden debajo de los megaproyectos.",
    "Las botas al revés fueron la señal. Los cuerpos tenían las botas al revés porque no eran guerrilleros, eran civiles inocentes a quienes vistieron con uniformes militares.",
    "Quienes llegaron al poder para matar jóvenes inocentes y pobres para hacerlos pasar por guerrilleros ahora utilizan a personas para limpiarse las manos untadas de sangre.",
    "Soy una madre que hoy día llora la falta de un hijo que las fuerzas del Gobierno me arrebataron a mí y que hoy llora sin ser escuchada.",
    "No ha sido un proceso sencillo, pues les exige volver sobre las heridas. Gradualmente, el ejercicio grupal, el compartir con otras personas y los talleres de apoyo psicológico les han ayudado a sanar.",
    "Las mujeres de MAFAPO buscan poder continuar, salir adelante, trabajar con la comunidad y permanecer en la búsqueda de la verdad y la justicia. Decidieron no atrincherarse en la tragedia.",
    "Ellas decidieron trascender su propia condición de víctimas, reclamando empatía con su causa más que lástima. La búsqueda de la justicia ya no es solo por sus familiares sino en representación de las demás víctimas.",
    "En repetidas ocasiones las han intentado callar con amenazas, pero no lo han logrado. Seguirán trabajando por la búsqueda de la verdad.",
    "Flor Hilda preguntó, con la voz entrecortada: ¿por qué le disparó? Pídale perdón a Dios. Mi hijo se fue pero sigue vivo en mi memoria. Clamamos justicia.",
    "Este hecho acabó con las fechas especiales y sus objetos son las memorias vivas de sus recuerdos. En 13 años no ha tenido una audiencia plena.",
    "Para estas mujeres, la búsqueda de justicia se ha transformado en fuerza para luchar. Se han apropiado de su dolor y lo han convertido en herramienta de resistencia y memoria.",
    "La verdad sigue amenazada por el negacionismo del poder. Decir que nuestros hijos no estarían recogiendo café solo refleja un enorme desprecio a la vida de las víctimas civiles.",
]

def main():
    import torch
    from transformers import AutoTokenizer, AutoModel

    # ── Cargar candidatos v5 ──────────────────────────────────────
    print("=" * 60)
    print("CFH — Centroide MAFAPO v5")
    print("=" * 60)

    with open(CANDIDATOS, encoding="utf-8") as f:
        data = json.load(f)
    candidatos = data["segmentos"]
    print(f"  Candidatos v5 cargados: {len(candidatos)}")
    print(f"  Por nivel: {data['por_nivel']}")

    # ── Cargar modelo en CPU ──────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
    print(f"\n  Cargando ConfliBERT-Spanish en {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  ✓ Modelo cargado")

    def get_emb(text):
        if not text or len(text.strip()) < 10:
            return np.zeros(768)
        inputs = tokenizer(text, return_tensors="pt", max_length=512,
                           truncation=True, padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    # ── Construir lista ponderada de textos+pesos ─────────────────
    # Base MAFAPO (texto escrito): w=1.0
    # Candidatos nivel 1-2 (voz directa identificada): w=1.8
    # Candidatos nivel 3 (señal mínima): w=1.0
    textos_pesos = []
    for t in TEXTOS_MAFAPO_BASE:
        textos_pesos.append((t, 1.0))
    for c in candidatos:
        w = 1.8 if c["nivel"] in (1, 2) else 1.0
        textos_pesos.append((c["texto"], w))

    n_total = len(textos_pesos)
    n_w18 = sum(1 for _, w in textos_pesos if w == 1.8)
    print(f"\n  Total textos para centroide v5: {n_total}")
    print(f"    Base MAFAPO (w=1.0): {len(TEXTOS_MAFAPO_BASE)}")
    print(f"    Voz directa N1-N2 (w=1.8): {n_w18}")
    print(f"    Resto N3 (w=1.0): {n_total - len(TEXTOS_MAFAPO_BASE) - n_w18}")

    # ── Calcular embeddings ───────────────────────────────────────
    print(f"\n  Calculando embeddings (CPU, ~{n_total} textos)...")
    embeddings = []
    pesos = []
    for i, (texto, w) in enumerate(textos_pesos):
        emb = get_emb(texto)
        if np.linalg.norm(emb) > 0:  # descartar vacíos
            embeddings.append(emb)
            pesos.append(w)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_total}")

    embeddings = np.array(embeddings)
    pesos = np.array(pesos)

    # ── Centroide ponderado ───────────────────────────────────────
    centroide_v5 = np.average(embeddings, axis=0, weights=pesos)

    print(f"\n  ✓ Centroide MAFAPO v5 calculado")
    print(f"    Shape: {centroide_v5.shape}")
    print(f"    Norma: {np.linalg.norm(centroide_v5):.3f}")
    print(f"    Textos efectivos: {len(embeddings)}")
    print(f"    Margen estimado: ±{100/(len(embeddings)**0.5):.1f}%")

    # ── Comparar con v4 ───────────────────────────────────────────
    v4_path = REF_DIR / "centroide_mafapo_v4.npy"
    if v4_path.exists():
        cent_v4 = np.load(v4_path)
        cos_sim = np.dot(centroide_v5, cent_v4) / (
            np.linalg.norm(centroide_v5) * np.linalg.norm(cent_v4))
        print(f"    Similitud coseno con v4: {cos_sim:.4f}")

    # ── Guardar ───────────────────────────────────────────────────
    out_path = REF_DIR / "centroide_mafapo_v5.npy"
    np.save(out_path, centroide_v5)
    print(f"\n  → Guardado: {out_path}")

    # Inventario
    inventario = {
        "version": "v5",
        "n_textos_total": int(len(embeddings)),
        "n_base_mafapo": len(TEXTOS_MAFAPO_BASE),
        "n_candidatos_v5": len(candidatos),
        "n_voz_directa_w18": int(n_w18),
        "margen_estimado_pct": round(100/(len(embeddings)**0.5), 2),
        "norma": float(np.linalg.norm(centroide_v5)),
        "modelo": model_name,
        "metodo": "CLS token, promedio ponderado w=1.8 voz directa / w=1.0 escrito",
    }
    inv_path = REF_DIR / "inventario_centroide_v5.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inventario, f, ensure_ascii=False, indent=2)
    print(f"  → Inventario: {inv_path}")
    print("\n[CFH] Centroide v5 completado.")


if __name__ == "__main__":
    main()

"""
CFH — Centroide MAFAPO v5 FINAL (objetivo ±5%)
================================================
Combina TRES fuentes de textos crudos de voz de víctimas, sin duplicar:
  1. 25 textos base MAFAPO (informe 2021)        — w=1.0
  2. Candidatos v4 (corpus_c_victimas_ampliado)  — w por nivel
  3. Candidatos v5 nuevos (corpus_c_victimas_v5) — w por nivel

Ponderación: w=1.8 voz directa (nivel 1-2), w=1.0 resto (nivel 3 + base escrito).
Deduplicación por los primeros 100 caracteres del texto.

Método idéntico al v4: embedding CLS, promedio ponderado, np.save.

Ejecutar (CPU local):
  conda activate cfh
  python code/cfh_centroide_mafapo_v5_final.py
"""
import json
import numpy as np
from pathlib import Path

REPO    = Path(__file__).resolve().parent.parent
REF_DIR = REPO / "data" / "referencias"
JSON_V4 = REF_DIR / "corpus_c_victimas_ampliado_v4.json"  # vacío — v4 ya incluido en v5
JSON_V5 = REF_DIR / "corpus_c_victimas_v5_clean.json"

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

def cargar_candidatos(ruta):
    """Carga segmentos de un JSON de candidatos; retorna lista de (texto, nivel)."""
    if not ruta.exists():
        print(f"  ! No existe: {ruta.name} — se omite")
        return []
    with open(ruta, encoding="utf-8") as f:
        data = json.load(f)
    segs = data.get("segmentos", [])
    return [(s["texto"], s.get("nivel", 3)) for s in segs]

def main():
    import torch
    from transformers import AutoTokenizer, AutoModel

    print("=" * 60)
    print("CFH — Centroide MAFAPO v5 FINAL (objetivo ±5%)")
    print("=" * 60)

    # ── Reunir textos de las 3 fuentes con deduplicación ──────────
    textos_pesos = []
    vistos = set()

    def agregar(texto, peso):
        clave = texto.strip()[:100]
        if clave and clave not in vistos and len(texto.strip()) >= 10:
            vistos.add(clave)
            textos_pesos.append((texto, peso))
            return True
        return False

    # 1. Base MAFAPO
    n_base = sum(agregar(t, 1.0) for t in TEXTOS_MAFAPO_BASE)

    # 2. Candidatos v4
    cands_v4 = cargar_candidatos(JSON_V4)

    n_v4 = sum(agregar(t, 1.8 if niv in (1, 2) else 1.0) for t, niv in cands_v4)

    # 3. Candidatos v5 nuevos
    cands_v5 = cargar_candidatos(JSON_V5)
    n_v5 = sum(agregar(t, 1.8 if niv in (1, 2) else 1.0) for t, niv in cands_v5)

    n_total = len(textos_pesos)
    n_w18 = sum(1 for _, w in textos_pesos if w == 1.8)
    print(f"\n  Textos únicos reunidos: {n_total}")
    print(f"    Base MAFAPO:        {n_base}")
    print(f"    Candidatos v4:      {n_v4} (de {len(cands_v4)})")
    print(f"    Candidatos v5:      {n_v5} (de {len(cands_v5)})")
    print(f"    Voz directa (w=1.8): {n_w18}")

    # ── Cargar modelo ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
    print(f"\n  Cargando ConfliBERT-Spanish en {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  ✓ Modelo cargado")

    def get_emb(text):
        inputs = tokenizer(text, return_tensors="pt", max_length=512,
                           truncation=True, padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    # ── Embeddings ────────────────────────────────────────────────
    print(f"\n  Calculando embeddings ({n_total} textos)...")
    embs, pesos = [], []
    for i, (texto, w) in enumerate(textos_pesos):
        emb = get_emb(texto)
        if np.linalg.norm(emb) > 0:
            embs.append(emb)
            pesos.append(w)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_total}")

    embs = np.array(embs)
    pesos = np.array(pesos)
    centroide = np.average(embs, axis=0, weights=pesos)

    n_eff = len(embs)
    margen = 100 / (n_eff ** 0.5)
    print(f"\n  ✓ Centroide MAFAPO v5 FINAL calculado")
    print(f"    Textos efectivos: {n_eff}")
    print(f"    Norma:            {np.linalg.norm(centroide):.3f}")
    print(f"    Margen estimado:  ±{margen:.1f}%")

    v4_path = REF_DIR / "centroide_mafapo_v4.npy"
    if v4_path.exists():
        cv4 = np.load(v4_path)
        cos = np.dot(centroide, cv4) / (np.linalg.norm(centroide) * np.linalg.norm(cv4))
        print(f"    Similitud coseno con v4: {cos:.4f}")

    # ── Guardar (sobrescribe el v5 anterior) ──────────────────────
    out = REF_DIR / "centroide_mafapo_v5.npy"
    np.save(out, centroide)
    print(f"\n  → Guardado: {out}")

    inventario = {
        "version": "v5_final",
        "n_textos_efectivos": int(n_eff),
        "n_base_mafapo": int(n_base),
        "n_candidatos_v4": int(n_v4),
        "n_candidatos_v5": int(n_v5),
        "n_voz_directa_w18": int(n_w18),
        "margen_estimado_pct": round(margen, 2),
        "norma": float(np.linalg.norm(centroide)),
        "modelo": model_name,
        "metodo": "CLS token, promedio ponderado w=1.8 voz directa / w=1.0 escrito, dedup por 100 chars",
    }
    with open(REF_DIR / "inventario_centroide_v5.json", "w", encoding="utf-8") as f:
        json.dump(inventario, f, ensure_ascii=False, indent=2)
    print(f"  → Inventario actualizado")
    print(f"\n[CFH] Centroide v5 FINAL completado. Margen: ±{margen:.1f}%")


if __name__ == "__main__":
    main()

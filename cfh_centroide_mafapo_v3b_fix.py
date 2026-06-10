"""
CFH — Centroide MAFAPO v3b (fix): reconstruye corpus completo desde cero
=========================================================================
El JSON de v3 solo tenía 32 textos porque los segmentos del Corpus C
se calcularon en memoria pero no se persistieron. Este script los
reconstruye completos, limpia los contaminados y agrega Dejusticia 2024.

Ejecutar:
  python -c "import sys; sys.path.insert(0,'code/src'); exec(open('cfh_centroide_mafapo_v3b_fix.py',encoding='utf-8').read())"
"""

import json, re
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cosine as cosine_dist

REPO    = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
REF_DIR = REPO / "data" / "referencias"
REF_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — Textos originales auditados (17 conservados de voz directa)
# ══════════════════════════════════════════════════════════════════════════
ORIGINALES = [
    "Con la mano en el alma, pido justicia. Que nos entreguen a la persona que realmente le segó la vida a mi hijo. Para mí es muy triste que en 13 años nadie haya respondido por la muerte de mi hijo.",
    "Tuve que vender mi casa para autofinanciar la búsqueda de mi hijo. Hasta el momento nadie sabe con certeza qué pasó con el joven.",
    "Es difícil pensar en acciones restaurativas cuando los principales responsables no han dicho la verdad y no han pedido perdón. El Ejército ha incumplido su promesa.",
    "No respetaron a las víctimas. El general tiene que aceptar esa verdad, que eso lo hizo él.",
    "Nuestros hijos no son una DUDA, la certeza de sus vidas no se relativiza, los parimos y el Estado los asesinó sin lugar a DUDAS.",
    "Si no fueran 6.402, así sean 50, sea uno, fueron crímenes de Estado y no se pueden quedar en el olvido.",
    "Sentimos como si hubieran tirado a nuestros hijos a una fosa común. Aún no nos reponemos. Nosotras seguimos sin encontrar justicia.",
    "Los cuerpos de nuestros familiares, encontrados como falsas bajas en combate en Ocaña, fueron alicorados y drogados antes de ser asesinados. Los trataron peor que a los animales.",
    "Los jóvenes de Soacha eran inocentes, trabajadores. El Ejército los reclutó con engaños prometiéndoles trabajo. Los llevaron a lugares que ellos no conocían y los ejecutaron.",
    "Mi hijo salió a buscar trabajo y nunca volvió. ¿Cómo podían volverse guerrilleros y enfrentarse en un combate contra el Ejército en tan pocas horas?",
    "Nos sumamos a esa intención de que en El Copey donde se cree que está el cuerpo del hijo de Doris no se construya ningún pavimento. Solicitamos que esos cuerpos no queden debajo de los megaproyectos.",
    "Las botas al revés fueron la señal. Los cuerpos tenían las botas al revés porque no eran guerrilleros, eran civiles inocentes a quienes vistieron con uniformes militares.",
    "Quienes llegaron al poder para matar jóvenes inocentes y pobres para hacerlos pasar por guerrilleros ahora utilizan a personas para limpiarse las manos untadas de sangre.",
    "Soy una madre que hoy día llora la falta de un hijo que las fuerzas del Gobierno me arrebataron a mí y que hoy llora sin ser escuchada.",
    "Flor Hilda preguntó, con la voz entrecortada: ¿por qué le disparó? Pídale perdón a Dios. Mi hijo se fue pero sigue vivo en mi memoria. Clamamos justicia.",
    "La verdad sigue amenazada por el negacionismo del poder. Decir que nuestros hijos no estarían recogiendo café solo refleja un enorme desprecio a la vida de las víctimas civiles.",
    # textos nuevos verificados del v3
    "Venimos ante esta Jurisdicción Especial para la Paz con la esperanza de que por fin se nos escuche. Llevamos diez años pidiendo justicia y la justicia ordinaria nos ha fallado.",
    "Pedimos garantías de seguridad para continuar trabajando. Nos han amenazado por buscar la verdad sobre la muerte de nuestros hijos.",
    "Queremos una verdad completa y profunda. No una verdad a medias que tape lo que pasó con nuestros hijos.",
    "Nadie nos dijo que nuestros hijos eran guerrilleros. El Ejército fue a nuestras casas y nos entregó los cuerpos vestidos con ropa que no era de ellos.",
    "Mi hijo me dijo que se iba a trabajar. Un hombre le prometió trabajo en el campo. Dos días después me llamaron a decirme que había muerto en combate. Mi hijo nunca había tenido un arma en la vida.",
    "Cuando fui a reconocer el cuerpo de mi hijo en Ocaña, tenía ropa militar que nunca le había visto. Las botas estaban al revés. Yo supe en ese momento que lo habían matado y lo habían disfrazado.",
    "No queremos plata. Queremos que digan la verdad. Queremos que digan quién dio la orden de matar a nuestros hijos y por qué.",
    "Nos propusimos hacer más bombo de lo normal para que Colombia conozca y dimensione nuestra tragedia. ¡Qué mejor que el arte para hacerlo!",
    "La necesidad de verdad y justicia, tras las ejecuciones extrajudiciales de 19 jóvenes, nos llevó a unirnos. Éramos mujeres cabeza de hogar dedicadas a cuidar nuestras familias.",
    "Recorrimos 640 kilómetros en bus para estar presentes. Esa distancia es nada comparada con los 13 años que llevamos buscando justicia.",
    "Hubo varias trabas de ministros anteriores para cumplir con este acto. Es un gran paso para demostrar que lo que pasó con nuestros familiares fueron crímenes de Estado, que no eran guerrilleros.",
    "No consideramos que esto deba ser un acto de excusas, sino que debe ser de perdón público, no solo para las madres de Soacha y Bogotá, es un perdón que se le debe a 6.402 madres.",
    "Es muy importante para nosotros dar a conocer esa cifra porque es la manera que estamos demostrando que sí fueron hechos reales y no casos aislados como se habló en el 2008.",
    "La cifra podría ser el doble. Lo que se hizo fue una práctica sistemática del Ejército. No eran guerrilleros, no era que estuvieran recogiendo café.",
    "Desde nuestros inicios procuramos nuestra propia recuperación emocional mientras luchamos por la verdad, la justicia, la reparación y las garantías de no repetición.",
    "Llevamos una batalla contra el olvido, dando visibilidad a estos terribles hechos y tejiendo vínculos con organizaciones defensoras de derechos humanos alrededor del mundo.",
]

# ══════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — Segmentos Corpus C (extraer de nuevo con filtro más estricto)
# ══════════════════════════════════════════════════════════════════════════
CORPUS_C_DIR = REPO / "corpus_c"

# Frases que indican lenguaje institucional JEP (NO voz de víctimas)
FRASES_INST = [
    "audiencia pública de reconocimiento",
    "ruta dialógica",
    "reconocimiento de verdad y responsabilidad",
    "determinación de los hechos",
    "aceptación de responsabilidad",
    "sala de reconocimiento",
    "jurisdicción especial",
    "esclarecimiento de la verdad",
    "negar los hechos o aducir",
]

# Frases que SÍ indican voz de familiares/víctimas
LEXICON_VICTIMAS = [
    "mi hijo", "mi hija", "mi hermano", "mi hermana", "mi madre", "mi padre",
    "nos lo mataron", "lo asesinaron", "lo mataron", "lo llevaron",
    "nunca volvió", "era inocente", "era civil", "no era guerrillero",
    "soy la mamá", "soy la hermana", "soy el padre", "soy hermano",
    "vengo por", "busco a", "pido justicia", "mi familia",
]

def es_voz_victima(texto):
    t = texto.lower()
    tiene_inst = any(f in t for f in FRASES_INST)
    tiene_vic  = any(f in t for f in LEXICON_VICTIMAS)
    return tiene_vic and not tiene_inst

segmentos_c = []
for jf in CORPUS_C_DIR.glob("*segments*.json"):
    try:
        with open(jf, encoding="utf-8") as f:
            datos = json.load(f)
        segs = datos if isinstance(datos, list) else next(
            (v for v in datos.values() if isinstance(v, list)), [])
        candidatos = []
        for s in segs:
            txt = s.get("text", s.get("texto", s.get("transcript", ""))) if isinstance(s, dict) else str(s)
            if len(txt) >= 40 and es_voz_victima(txt):
                candidatos.append(txt)
        # top 4 por audiencia (más estricto que antes)
        segmentos_c.extend(candidatos[:4])
        print(f"  {jf.name}: {len(segs)} segs → {len(candidatos)} víctimas → tomados {min(4,len(candidatos))}")
    except Exception as e:
        print(f"  ✗ {jf.name}: {e}")

print(f"  Total segmentos Corpus C (filtro estricto): {len(segmentos_c)}")

# ══════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — Textos Dejusticia 2024 (15 testimonios directos)
# ══════════════════════════════════════════════════════════════════════════
DEJUSTICIA = [
    "Una madre conoce a su hijo. Yo sabía que ese era mi hijo, pero que lo estaban haciendo pasar por algo que no era: un guerrillero.",
    "Mami, todo lo que voy a hacer es por usted, para que no trabaje tanto y mis hermanas estén tranquilas.",
    "Lo único que puede hacer es cuidar a mi mamá, y dígale que no le voy a poder cumplir la promesa que le hice.",
    "Solo ellas saben el dolor que es perder a un hijo de esta manera tan cruel. Hay peleas y disgustos, pero ellas son para mí, mi familia.",
    "La vida nos puso ahí, y nos tocó asumir ese rol.",
    "El dolor ya no solo era procesar la muerte de nuestros hijos, sino también el sufrimiento de ver cómo manchaban sus nombres, cómo se les despojaba de su honra incluso en la muerte.",
    "Sabíamos que era una mentira, que esa acusación no tenía sentido. Quienes debían cuidarlos fueron los que los mataron.",
    "El cuerpo de Eduardo llevaba unas botas puestas al revés y una mochila colgada al hombro. Pero yo sabía que él nunca usaba botas. Siempre se vestía de manera formal.",
    "Para encontrar a su hermano, tuvo que mover uno a uno los cuerpos. Daniel era el último de la pila.",
    "Sentíamos un alivio por haberlos hallado. Pero ese alivio estaba teñido de una tristeza profunda, pues las personas que más amábamos estaban muertas.",
    "Gloria, debilitada por la depresión y por ataques de asma, no se sintió capaz de hacer el viaje sola a identificar el cuerpo de su hijo.",
    "Necesitábamos la verdad. Queríamos mostrarle al país que nuestros hijos no eran guerrilleros, que eran inocentes.",
    "Para mí era muy difícil pensar que quienes debían cuidar a nuestros hijos fueron los que los mataron.",
    "Cuando el país nos ve con nuestras botas pintadas de colores vibrantes, sabe que ahí estamos, recordando las injusticias y resistiendo con fuerza.",
    "Sabemos que nuestros hijos, donde sea que estén, están muy orgullosos de todo lo que hemos ayudado.",
]

# ══════════════════════════════════════════════════════════════════════════
# CORPUS FINAL
# ══════════════════════════════════════════════════════════════════════════
TODOS_TEXTOS = ORIGINALES + segmentos_c + DEJUSTICIA

print(f"\n{'='*55}")
print(f"CORPUS MAFAPO v3b — RECONSTRUIDO")
print(f"{'='*55}")
print(f"  Originales auditados:    {len(ORIGINALES)}")
print(f"  Corpus C (filtro):       {len(segmentos_c)}")
print(f"  Dejusticia 2024:         {len(DEJUSTICIA)}")
print(f"  TOTAL:                   {len(TODOS_TEXTOS)}")
print(f"  Meta 75: {'✓ ALCANZADA' if len(TODOS_TEXTOS)>=75 else f'⚠ Faltan {75-len(TODOS_TEXTOS)}'}")

# ══════════════════════════════════════════════════════════════════════════
# CALCULAR CENTROIDE
# ══════════════════════════════════════════════════════════════════════════
import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Dispositivo: {DEVICE}")
MODEL_NAME = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("  ✓ Modelo cargado")

def get_emb(texto):
    if not texto or len(texto.strip()) < 10:
        return np.zeros(768)
    inp = tokenizer(texto, return_tensors="pt", max_length=512,
                    truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inp)
    return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

print(f"\n  Calculando centroide MAFAPO v3b ({len(TODOS_TEXTOS)} textos)...")
embs = []
for i, t in enumerate(TODOS_TEXTOS):
    embs.append(get_emb(t))
    if (i+1) % 10 == 0:
        print(f"    {i+1}/{len(TODOS_TEXTOS)}...")
centroide_v3b = np.mean(embs, axis=0)
print(f"  ✓ Centroide v3b: norma={np.linalg.norm(centroide_v3b):.3f}")

np.save(str(REF_DIR / "centroide_mafapo_v3b.npy"), centroide_v3b)

# CIDH desde disco
cidh_path = REF_DIR / "centroide_cidh_v3.npy"
if cidh_path.exists():
    centroide_cidh = np.load(str(cidh_path))
    print("  ✓ Centroide CIDH cargado desde disco")
else:
    print("  ⚠ centroide_cidh_v3.npy no encontrado — recalcular manualmente")
    exit()

# ══════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN CORRELACIÓN
# ══════════════════════════════════════════════════════════════════════════
dist_polos = cosine_dist(centroide_v3b, centroide_cidh)

PROCESSED_DIR = REPO / "data" / "processed"
textos_muestra = []
for txt in list(PROCESSED_DIR.rglob("*.txt"))[:50]:
    try:
        c = txt.read_text(encoding="utf-8", errors="ignore")
        pp = [p.strip() for p in c.split("\n\n") if 200 <= len(p.strip()) <= 500]
        textos_muestra.extend(pp[:3])
    except: pass
textos_muestra = textos_muestra[:100]

d_maf, d_cid = [], []
for t in textos_muestra:
    e = get_emb(t)
    d_maf.append(cosine_dist(e, centroide_v3b))
    d_cid.append(cosine_dist(e, centroide_cidh))

from scipy.stats import pearsonr, spearmanr
r_p, p_p = pearsonr(d_maf, d_cid)
r_s, p_s = spearmanr(d_maf, d_cid)

reporte = f"""
REPORTE CORRELACIÓN y₈ vs y₉ — CFH v3b (reconstruido)
========================================================
Fecha: {datetime.now().isoformat()}
N muestra: {len(textos_muestra)} bloques

Corpus MAFAPO v3b: {len(TODOS_TEXTOS)} textos
  - {len(ORIGINALES)} originales auditados (voz directa)
  - {len(segmentos_c)} segmentos Corpus C (filtro estricto)
  - {len(DEJUSTICIA)} testimonios Dejusticia 2024

Distancia entre polos MAFAPO vs CIDH:
  v2  (original): ~0.14
  v3  (anterior):  0.0919
  v3b (actual):    {dist_polos:.4f}  {'↑ mejoró diferenciación' if dist_polos > 0.0919 else '— similar'}

Correlación y₈ vs y₉ (Pearson):
  v2:  r ≈ 0.86  (problema original)
  v3:  r = 0.7316 (meta alcanzada)
  v3b: r = {r_p:.4f}  (p={p_p:.4f})
  Spearman ρ = {r_s:.4f}  (p={p_s:.4f})

Meta r < 0.80: {'✓ ALCANZADA' if r_p < 0.80 else '⚠ NO ALCANZADA'}
{'✓ Mejora respecto a v3' if r_p < 0.7316 else '— Sin mejora adicional respecto a v3'}
"""

print(reporte)
with open(str(REF_DIR / "reporte_correlacion_v3b.txt"), "w", encoding="utf-8") as f:
    f.write(reporte)

# Guardar inventario
inventario = {
    "version": "v3b",
    "timestamp": datetime.now().isoformat(),
    "resumen": {
        "total": len(TODOS_TEXTOS),
        "originales": len(ORIGINALES),
        "corpus_c": len(segmentos_c),
        "dejusticia_2024": len(DEJUSTICIA),
        "pearson_r": round(r_p, 4),
        "meta_alcanzada": r_p < 0.80
    }
}
with open(str(REF_DIR / "corpus_mafapo_v3b.json"), "w", encoding="utf-8") as f:
    json.dump(inventario, f, ensure_ascii=False, indent=2)

print(f"\n[CFH] v3b completado. Archivos en data/referencias/")
print(f"  centroide_mafapo_v3b.npy")
print(f"  corpus_mafapo_v3b.json")
print(f"  reporte_correlacion_v3b.txt")

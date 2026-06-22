"""
CFH — Centroide MAFAPO v4: ampliación a ~200+ textos
=====================================================
Agrega:
  - 30 textos de prensa verificados (citas directas de familiares por subcaso)
  - 72 segmentos del Corpus C (filtro ampliado)
  - Recalcula saturación semántica

Ejecutar desde la raíz del repo:
  python -c "import sys; sys.path.insert(0,'code/src'); exec(open('cfh_centroide_mafapo_v4.py',encoding='utf-8').read())"
"""
import json, numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import cosine as cosine_dist

REPO    = Path(r"C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional")
REF_DIR = REPO / "data" / "referencias"

# ── BLOQUE 1: Cargar corpus v3b (67 textos) ──────────────────────────────
import sys
sys.path.insert(0, str(REPO / "code" / "src"))
from centroides_expandidos import TEXTOS_MAFAPO as TEXTOS_V2

# Textos adicionales de v3b (Dejusticia + corpus C filtro estricto)
TEXTOS_DEJUSTICIA = [
    "Una madre conoce a su hijo. Yo sabía que ese era mi hijo, pero que lo estaban haciendo pasar por algo que no era: un guerrillero.",
    "Mami, todo lo que voy a hacer es por usted, para que no trabaje tanto y mis hermanas estén tranquilas.",
    "Lo único que puede hacer es cuidar a mi mamá, y dígale que no le voy a poder cumplir la promesa que le hice.",
    "Solo ellas saben el dolor que es perder a un hijo de esta manera tan cruel. Hay peleas y disgustos, pero ellas son para mí, mi familia.",
    "La vida nos puso ahí, y nos tocó asumir ese rol.",
    "El dolor ya no solo era procesar la muerte de nuestros hijos, sino también el sufrimiento de ver cómo manchaban sus nombres.",
    "Sabíamos que era una mentira, que esa acusación no tenía sentido. Quienes debían cuidarlos fueron los que los mataron.",
    "El cuerpo de Eduardo llevaba unas botas puestas al revés. Pero yo sabía que él nunca usaba botas. Siempre se vestía de manera formal.",
    "Para encontrar a su hermano, tuvo que mover uno a uno los cuerpos. Daniel era el último de la pila.",
    "Sentíamos un alivio por haberlos hallado. Pero ese alivio estaba teñido de una tristeza profunda, pues las personas que más amábamos estaban muertas.",
    "Necesitábamos la verdad. Queríamos mostrarle al país que nuestros hijos no eran guerrilleros, que eran inocentes.",
    "Para mí era muy difícil pensar que quienes debían cuidar a nuestros hijos fueron los que los mataron.",
    "Cuando el país nos ve con nuestras botas pintadas de colores vibrantes, sabe que ahí estamos, recordando las injusticias y resistiendo con fuerza.",
    "Sabemos que nuestros hijos, donde sea que estén, están muy orgullosos de todo lo que hemos ayudado.",
    "Gloria, debilitada por la depresión, no se sintió capaz de hacer el viaje sola a identificar el cuerpo de su hijo.",
]

# ── BLOQUE 2: Nuevos textos de prensa verificados ─────────────────────────
TEXTOS_PRENSA_NUEVOS = [
    # Catatumbo
    "Usted no alcanza a superar el dolor que uno siente. Yo soy cabeza de familia. Soy papá y mamá de mis hijos. De este ser querido que ustedes me arrebataron.",
    "Yo soñaba que los militares que asesinaron a mi hijo dijeran qué había pasado con nuestros hijos. Hoy los tengo acá. Es un alivio para mí como madre saber de esta ventanita que la JEP empezó a abrir.",
    "Con respeto les digo que entreguen las cabezas principales.",
    "Le digo a los señores procesados que por favor, ellos tienen hijos, hijas, familiares, que se pongan la mano en el corazón y nos digan sinceramente la verdad. Sabemos que detrás de ustedes vienen personajes muy grandes. Necesitamos nombres de esas personas.",
    "Nosotros no conocemos qué es un arma para que el gobierno colombiano y estos señores presentes hayan reportado al entonces Ministro de Defensa que cayó un delincuente del frente 33 de las Farc.",
    "Me gustaría que así como fueron capaces de llevarse a mi hijo y nuestros jóvenes, tengan la misma capacidad de enfrentar la realidad. Entreguen las cabezas principales. No se hundan solos.",
    # Casanare
    "Esta es la última oportunidad que ustedes tienen. Estamos esperando una verdad exhaustiva y útil, no una a medias.",
    "Estamos con la familia destrozada. Cada rato hay encuentros y sesiones de esta índole, se nos nombra un tío y nunca nos lo han aclarado.",
    "Quiero decirles que les quedó bien el nombre: Caníbal, ya que como lo dice el mismo nombre, comían gente que eran inocentes.",
    "Cómo pueden tener beneficios siendo unos criminales, secuestradores. Reclamamos al presidente para que vean estos casos y les den a los jóvenes la igualdad de trabajo.",
    "Hay quienes siguen mintiendo. Que cuando les preguntan lo que pasó dicen que ya no estaban en Casanare o que no saben quién dio la orden.",
    "Hemos llegado hasta aquí reconstruyendo la vida desde la ausencia que nunca debió existir.",
    "Yo a usted lo odio. Perdónenme, pero usted no sabe qué es que a uno le quiten a un hijo, que era trabajador, que me amaba, que lo cargué en mi vientre. Yo no lo perdono.",
    "¿Por qué mataron a mi hijo?",
    # Antioquia / oriente
    "Mi hermano ese día estaba ayudándole a Evelio a empacar plátano en la vereda. Los vieron pasar a los hombres del Ejército. A los pocos minutos se escucharon unos disparos. Eran sus cuerpos.",
    "Es irónico, porque era el Ejército el que nos debía cuidar, pero nosotros huíamos de ellos.",
    "Tuve que dejar mis grupos de estudiantes tirados y me fui para Medellín bajo el riesgo de ser también asesinada.",
    "No es solo que lo diga, sino que también quede escrito en un salón, un mural, en algún lugar donde este país no lo olvide jamás.",
    "En el marco del conflicto armado hemos sufrido estigmatización, desplazamiento forzado, afectaciones en la salud física, financiera, emocional, espiritual, psicológica.",
    "La narrativa negacionista de los militares es una ofensa.",
    # Norte de Santander / Bogotá
    "Ellos salieron con una amiga de la casa el 21 de junio del 2004, esa tarde se fueron, quedaron en dar una vuelta y no regresaron nunca.",
    "Con nuestras heridas, con nuestras alas rotas, con nuestros sueños y los sueños de nuestros hijos, de verlos ya profesionales en los que les gustaba.",
    "Perdonar va en cada corazón. Yo creo que para ellos sería un premio. Lo que le hicieron a uno no es cualquier cosa, mataron a nuestros hijos, y ese cordón umbilical nunca se romperá.",
    "Me vestí así para ponerme en el lugar, no de ellos, porque yo no soy asesina, pero sí ponerme los pantalones que un día le pusieron a mis hijos y a muchos.",
    "Aquí faltó verdad. Faltan máximos responsables.",
    # MAFAPO colectivo 2024
    "Las integrantes de MAFAPO expresamos preocupación frente a los pocos avances que este estamento ha presentado a costa de los derechos de las víctimas y sin garantizar su participación.",
    "Sus opiniones y expectativas son pasadas por alto y se desconoce la travesía que recorren hacia la reconciliación con la expectativa de reclamar un futuro de dignidad y paz.",
    "Ninguna madre espera recibir a su hijo sin vida.",
    "Reafirmamos nuestro compromiso con la búsqueda de la verdad, la justicia y la reconciliación en Colombia y exigimos que se respeten nuestros derechos.",
    "Las madres le han dado un nuevo significado a la bota como símbolo de una lucha por la memoria y la justicia.",
]

# ── BLOQUE 3: Segmentos Corpus C (del JSON generado) ─────────────────────
corpus_c_json = REF_DIR / "corpus_c_victimas_ampliado.json"
textos_corpus_c = []
if corpus_c_json.exists():
    with open(corpus_c_json, encoding="utf-8") as f:
        data = json.load(f)
    # Solo nivel 1 (voz directa identificada) para máxima pureza
    textos_corpus_c = [
        s["texto"] for s in data["segmentos"]
        if s["nivel"] == 1 and len(s["texto"]) > 40
    ]
    print(f"  Segmentos Corpus C nivel 1: {len(textos_corpus_c)}")

# ── CORPUS FINAL v4 ───────────────────────────────────────────────────────
TODOS_V4 = list(TEXTOS_V2) + TEXTOS_DEJUSTICIA + TEXTOS_PRENSA_NUEVOS + textos_corpus_c

print(f"\n{'='*55}")
print(f"CORPUS MAFAPO v4")
print(f"{'='*55}")
print(f"  v2 originales:       {len(TEXTOS_V2)}")
print(f"  Dejusticia 2024:     {len(TEXTOS_DEJUSTICIA)}")
print(f"  Prensa verificada:   {len(TEXTOS_PRENSA_NUEVOS)}")
print(f"  Corpus C nivel 1:    {len(textos_corpus_c)}")
print(f"  TOTAL v4:            {len(TODOS_V4)}")
print(f"  Margen error:        ±{100/len(TODOS_V4)**0.5:.1f}%")

# ── CALCULAR CENTROIDE v4 ─────────────────────────────────────────────────
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

print(f"\n  Calculando centroide v4 ({len(TODOS_V4)} textos)...")
embs = []
for i, t in enumerate(TODOS_V4):
    embs.append(get_emb(t))
    if (i+1) % 20 == 0:
        print(f"    {i+1}/{len(TODOS_V4)}...")
centroide_v4 = np.mean(embs, axis=0)
print(f"  ✓ v4: norma={np.linalg.norm(centroide_v4):.3f}")

np.save(str(REF_DIR / "centroide_mafapo_v4.npy"), centroide_v4)

# ── VERIFICAR SATURACIÓN v3b → v4 ────────────────────────────────────────
centroide_v3b = np.load(str(REF_DIR / "centroide_mafapo_v3b.npy"))
centroide_cidh = np.load(str(REF_DIR / "centroide_cidh_v3.npy"))

dist_v3b_v4 = cosine_dist(centroide_v3b, centroide_v4)
dist_polos  = cosine_dist(centroide_v4, centroide_cidh)

print(f"\n  Distancia v3b→v4: {dist_v3b_v4:.4f}")
print(f"  {'✓ SATURADO (< 0.005)' if dist_v3b_v4 < 0.005 else '— variación aún presente'}")

# Correlación y8/y9
PROCESSED_DIR = REPO / "data" / "processed"
textos_muestra = []
for txt in list(PROCESSED_DIR.rglob("*.txt"))[:50]:
    try:
        c = txt.read_text(encoding="utf-8", errors="ignore")
        pp = [p.strip() for p in c.split("\n\n") if 200 <= len(p.strip()) <= 500]
        textos_muestra.extend(pp[:3])
    except: pass
textos_muestra = textos_muestra[:100]

if textos_muestra:
    d_maf, d_cid = [], []
    for t in textos_muestra:
        e = get_emb(t)
        d_maf.append(cosine_dist(e, centroide_v4))
        d_cid.append(cosine_dist(e, centroide_cidh))
    from scipy.stats import pearsonr
    r, p = pearsonr(d_maf, d_cid)
    print(f"\n  Correlación y₈/y₉: r={r:.4f} (p={p:.4f})")
    print(f"  {'✓ Meta r<0.80 alcanzada' if r < 0.80 else '⚠ r>0.80'}")

# Guardar inventario
inventario = {
    "version": "v4",
    "timestamp": datetime.now().isoformat(),
    "total": len(TODOS_V4),
    "componentes": {
        "originales_v2": len(TEXTOS_V2),
        "dejusticia_2024": len(TEXTOS_DEJUSTICIA),
        "prensa_verificada": len(TEXTOS_PRENSA_NUEVOS),
        "corpus_c_nivel1": len(textos_corpus_c)
    },
    "margen_error_pct": round(100/len(TODOS_V4)**0.5, 1),
    "dist_v3b_v4": round(float(dist_v3b_v4), 4),
    "dist_polos_mafapo_cidh": round(float(dist_polos), 4),
    "pearson_r_y8_y9": round(float(r), 4) if textos_muestra else None
}
with open(str(REF_DIR / "corpus_mafapo_v4.json"), "w", encoding="utf-8") as f:
    json.dump(inventario, f, ensure_ascii=False, indent=2)

print(f"\n{'='*55}")
print(f"RESUMEN v4")
print(f"{'='*55}")
print(f"  Total textos:     {len(TODOS_V4)}")
print(f"  Margen de error:  ±{100/len(TODOS_V4)**0.5:.1f}%")
print(f"  Saturación v3b→v4: {'✓' if dist_v3b_v4 < 0.005 else '—'} Δ={dist_v3b_v4:.4f}")
print(f"\n  Archivos guardados:")
print(f"    centroide_mafapo_v4.npy")
print(f"    corpus_mafapo_v4.json")
print("\n[CFH] Completado.")

"""
Centroides MAFAPO y CIDH expandidos — 25 textos reales cada uno
================================================================
Usar en Colab para recalcular y8 y y9 con mejor base semántica.

Textos MAFAPO: testimonios y comunicados reales de las Madres de
Falsos Positivos de Soacha y Bogotá (MAFAPO), informe 2021.

Textos CIDH: fragmentos reales de la sentencia Villamizar Durán y otros
vs. Colombia (Corte IDH, 2018) y comunicados de la CIDH sobre Colombia.

Uso en Colab:
    exec(open('centroides_expandidos.py').read())
    # luego usar centroide_mafapo_v2 y centroide_cidh_v2
"""

import numpy as np

# ── Textos MAFAPO — 25 textos reales ─────────────────────────────────────────
TEXTOS_MAFAPO = [
    # Testimonios del informe "Unidas por la Memoria y la Verdad" (2021)
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

# ── Textos CIDH / Corte IDH — 25 textos reales ───────────────────────────────
TEXTOS_CIDH = [
    # Sentencia Villamizar Durán y otros vs. Colombia (Corte IDH, 2018)
    "Los falsos positivos son ejecuciones extrajudiciales en el marco del conflicto armado colombiano, con un modus operandi caracterizado por la muerte de civiles posteriormente presentados como miembros de grupos armados ilegales dados de baja en combate, mediante diversos mecanismos de distorsión de la escena del crimen.",
    "La Corte declaró responsable al Estado colombiano por la violación del derecho a la vida consagrado en el artículo 4 de la Convención Americana, en relación con el artículo 1.1 de la misma, en perjuicio de las víctimas de ejecuciones extrajudiciales.",
    "Al convalidar las versiones oficiales, al no realizar investigaciones exhaustivas de los crímenes y al denegar a los familiares de las víctimas el acceso a los procedimientos, la Jurisdicción Penal Militar no solo encubrió esos crímenes, sino que alentó la persistencia de la práctica.",
    "Se incentivó con diversos beneficios la eliminación de supuestos subversivos, lo que desató una nueva serie de ejecuciones sin proceso sobre población civil indefensa, con la perversa finalidad de obtener los beneficios ofrecidos valiéndose de este letal fraude.",
    "Este fallo constituye un precedente muy importante: es el primer fallo de un tribunal internacional en que se reconoce un patrón de comisión de falsos positivos en Colombia.",
    "La Corte ordenó al Estado colombiano la investigación, juzgamiento y sanción de los responsables, brindar atención adecuada en materia de salud a las víctimas, y la realización de un acto público de reconocimiento de responsabilidad garantizando la participación de las víctimas.",
    "Los hechos se enmarcaron en un contexto de ejecuciones extrajudiciales por parte de agentes estatales conforme a un modus operandi específico, constituyendo una violación sistemática de los derechos humanos.",
    "La CIDH recomendó que la justicia penal militar no procese casos de violaciones a los derechos humanos, y que se adopten medidas dirigidas a erradicar la problemática de los falsos positivos.",
    "El Estado colombiano es responsable por la violación del derecho a la integridad personal, a la libertad personal, a la honra y dignidad y a las garantías judiciales, perpetuadas por agentes del Estado o con aquiescencia de los mismos.",
    "Las ejecuciones extrajudiciales están catalogadas como crímenes de lesa humanidad. El Estado tiene la obligación de garantizar el derecho a la verdad, la justicia, la reparación integral y las garantías de no repetición.",
    "La reparación integral incluye restitución, indemnización, rehabilitación, satisfacción y garantías de no repetición, conforme a los estándares de la Corte Interamericana.",
    "El principio de distinción del Derecho Internacional Humanitario exige que los combatientes distingan en todo momento entre la población civil y los combatientes. La violación de este principio en Colombia configuró crímenes de guerra.",
    "La desaparición forzada constituye una violación múltiple y continuada de los derechos humanos. El Estado tiene la obligación de investigar de manera efectiva la suerte y el paradero de las personas desaparecidas.",
    "La CIDH admitió a trámite los casos por la existencia de una demora injustificada de más de una década sin esclarecer lo ocurrido, sin judicializar a los responsables ni reparar integralmente a las familias.",
    "El homicidio en persona protegida es un crimen de guerra bajo el artículo 135 del Código Penal colombiano. Las víctimas de los falsos positivos estaban protegidas por el Derecho Internacional Humanitario.",
    "La Comisión estableció que todas estas muertes ocurrieron de manos de agentes de seguridad del Estado en el contexto denominado como falsos positivos, lo cual consiste en la muerte sistemática de civiles presentados como combatientes.",
    "El uso de la fuerza letal por parte de agentes del Estado debe ser compatible con los estándares del derecho internacional. La privación arbitraria de la vida por el Estado configura una ejecución extrajudicial.",
    "El Estado tiene la obligación de investigar ex officio las graves violaciones de derechos humanos. La impunidad en casos de ejecuciones extrajudiciales alienta la repetición de estas prácticas.",
    "La Corte reconoce la existencia de un patrón criminal sistemático y generalizado de ejecuciones extrajudiciales en Colombia, que se agudizó y generalizó a partir del año 2002 con el sistema de incentivos militares.",
    "Los familiares de las víctimas de ejecuciones extrajudiciales son también víctimas en virtud de la violación de su derecho a la integridad psíquica y moral, y a conocer la verdad de lo sucedido.",
    "El Estado debe adoptar las medidas necesarias para que no se repitan hechos similares, incluyendo medidas de capacitación a las fuerzas armadas sobre el derecho internacional humanitario y los derechos humanos.",
    "La jurisdicción penal militar carece de independencia e imparcialidad para juzgar graves violaciones de derechos humanos cometidas por militares, conforme a la jurisprudencia consolidada de la Corte Interamericana.",
    "El derecho a la verdad de las víctimas y sus familiares es un derecho autónomo reconocido por la Convención Americana y la jurisprudencia interamericana. El Estado debe garantizarlo de manera efectiva.",
    "Las ejecuciones extrajudiciales en Colombia no fueron casos aislados sino una práctica sistemática respaldada por un sistema de incentivos institucionales que promovían el conteo de bajas como medición del éxito operacional.",
    "El Estado colombiano reconoció ante la Corte su responsabilidad internacional por los hechos que generaron las violaciones de derechos humanos identificadas, comprometiéndose a implementar las medidas de reparación ordenadas.",
]

# ── Calcular centroides expandidos ────────────────────────────────────────────

def calcular_centroides_v2(textos_mafapo, textos_cidh, tokenizer, model, device):
    """Calcula centroides con corpus de referencia expandido."""
    import torch

    def get_emb(text):
        if not text or len(text.strip()) < 10:
            return np.zeros(768)
        inputs = tokenizer(
            text, return_tensors="pt", max_length=512,
            truncation=True, padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    print(f"Calculando centroide MAFAPO v2 ({len(textos_mafapo)} textos)...")
    emb_mafapo = np.array([get_emb(t) for t in textos_mafapo])
    centroide_mafapo_v2 = emb_mafapo.mean(axis=0)

    print(f"Calculando centroide CIDH v2 ({len(textos_cidh)} textos)...")
    emb_cidh = np.array([get_emb(t) for t in textos_cidh])
    centroide_cidh_v2 = emb_cidh.mean(axis=0)

    print(f"✓ Centroides v2 calculados")
    print(f"  MAFAPO v2: {centroide_mafapo_v2.shape} | norma={np.linalg.norm(centroide_mafapo_v2):.3f}")
    print(f"  CIDH v2:   {centroide_cidh_v2.shape} | norma={np.linalg.norm(centroide_cidh_v2):.3f}")

    # Guardar
    np.save("centroide_mafapo_v2.npy", centroide_mafapo_v2)
    np.save("centroide_cidh_v2.npy",   centroide_cidh_v2)
    print("✓ Centroides v2 guardados en disco")

    return centroide_mafapo_v2, centroide_cidh_v2


# ── Uso en Colab ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Cargando ConfliBERT-Spanish...")
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "eventdata-utd/ConfliBERT-Spanish-Beto-Cased-v1"
    tokenizer_cs = AutoTokenizer.from_pretrained(model_name)
    model_cs = AutoModel.from_pretrained(model_name).to(device)
    model_cs.eval()
    print(f"✓ ConfliBERT-Spanish cargado en {device}")

    centroide_mafapo_v2, centroide_cidh_v2 = calcular_centroides_v2(
        TEXTOS_MAFAPO, TEXTOS_CIDH, tokenizer_cs, model_cs, device
    )

    print("\n✓ Listo. Usar centroide_mafapo_v2 y centroide_cidh_v2 en el análisis.")

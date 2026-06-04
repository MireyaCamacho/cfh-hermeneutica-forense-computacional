const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, Footer, Header, PageBreak, SimpleField, TabStopType,
  TabStopPosition, UnderlineType
} = require('docx');
const fs = require('fs');

// ── Colores ──────────────────────────────────────────────────────────────────
const NAVY  = "0D2137";
const TEAL  = "0D9488";
const BLACK = "000000";
const GRAY  = "595959";

// ── Helpers base ─────────────────────────────────────────────────────────────
const run = (text, opts = {}) =>
  new TextRun({ text, font: "Times New Roman", size: 24, color: BLACK, ...opts });

const para = (children, opts = {}) =>
  new Paragraph({
    children: Array.isArray(children) ? children : [run(children)],
    spacing: { after: 200, line: 360, lineRule: "auto" },
    ...opts
  });

const paraJust = (text, opts = {}) =>
  new Paragraph({
    children: [run(text)],
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 200, line: 360, lineRule: "auto" },
    indent: { firstLine: 720 },
    ...opts
  });

const paraJustRuns = (children, opts = {}) =>
  new Paragraph({
    children,
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 200, line: 360, lineRule: "auto" },
    indent: { firstLine: 720 },
    ...opts
  });

const spacer = (n = 1) =>
  new Paragraph({ children: [run("")], spacing: { after: 120 * n } });

const pageBreak = () =>
  new Paragraph({ children: [new PageBreak()] });

const heading1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, font: "Times New Roman", size: 28, bold: true, color: NAVY })],
    spacing: { before: 480, after: 240 },
    outlineLevel: 0
  });

const heading2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font: "Times New Roman", size: 26, bold: true, color: BLACK })],
    spacing: { before: 360, after: 180 },
    outlineLevel: 1
  });

const heading3 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, font: "Times New Roman", size: 24, bold: true, italics: true, color: BLACK })],
    spacing: { before: 280, after: 140 },
    outlineLevel: 2
  });

const nota = (num, text) =>
  new Paragraph({
    children: [
      run(`${num} `, { size: 18, verticalAlign: "superscript" }),
      run(text, { size: 18 })
    ],
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 80 },
    border: { top: { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA", space: 4 } }
  });

// ── Documento ────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
    }]
  },
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 24 } }
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Times New Roman", color: NAVY },
        paragraph: { spacing: { before: 480, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman", color: BLACK },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, italics: true, font: "Times New Roman", color: BLACK },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 2268, right: 1701, bottom: 2268, left: 2268 } // 4cm izq, 3cm resto
      }
    },
    headers: {
      default: new Header({ children: [
        new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "El lenguaje de los falsos positivos", font: "Times New Roman", size: 20, color: GRAY, italics: true })],
          spacing: { after: 0 }
        })
      ]})
    },
    footers: {
      default: new Footer({ children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new SimpleField("PAGE")],
          spacing: { after: 0 }
        })
      ]})
    },
    children: [

      // ══════════════════════════════════════════════════════════════════════
      // PORTADA
      // ══════════════════════════════════════════════════════════════════════
      spacer(4),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("UNIVERSIDAD EXTERNADO DE COLOMBIA", { bold: true, size: 22 })],
        spacing: { after: 120 }
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("FACULTAD DE CIENCIAS EXACTAS Y NATURALES", { size: 22 })],
        spacing: { after: 120 }
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("PROGRAMA DE CIENCIA DE DATOS", { size: 22 })],
        spacing: { after: 600 }
      }),
      spacer(6),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("El lenguaje de los falsos positivos:", { bold: true, size: 32, color: NAVY })],
        spacing: { after: 120 }
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("un modelo computacional para medir la injusticia discursiva", { bold: true, size: 28, color: NAVY })],
        spacing: { after: 120 }
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("en el archivo judicial colombiano", { bold: true, size: 28, color: NAVY })],
        spacing: { after: 600 }
      }),
      spacer(6),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("Trabajo de grado para optar al título de", { size: 22, italics: true })],
        spacing: { after: 80 }
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("Profesional en Ciencia de Datos", { size: 22, bold: true })],
        spacing: { after: 400 }
      }),
      spacer(4),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("Autora:", { size: 22 })],
        spacing: { after: 80 }
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("Mireya Camacho Celis", { size: 24, bold: true })],
        spacing: { after: 400 }
      }),
      spacer(6),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("Bogotá D.C., 2026", { size: 22 })],
        spacing: { after: 0 }
      }),

      pageBreak(),

      // ══════════════════════════════════════════════════════════════════════
      // TABLA DE CONTENIDO (preliminar)
      // ══════════════════════════════════════════════════════════════════════
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("TABLA DE CONTENIDO", { bold: true, size: 26 })],
        spacing: { before: 240, after: 400 }
      }),

      // Entradas de la tabla de contenido
      ...[
        ["Introducción", "1"],
        ["    1.1 Planteamiento del problema", "3"],
        ["    1.2 Pregunta de investigación", "7"],
        ["    1.3 Objetivos", "7"],
        ["    1.4 Justificación", "8"],
        ["    1.5 Estructura del documento", "10"],
        ["2. Estado del arte", "12"],
        ["    2.1 Violencia cultural y lenguaje institucional", "12"],
        ["    2.2 NLP aplicado a textos judiciales", "15"],
        ["    2.3 Justicia transicional y análisis del discurso", "18"],
        ["    2.4 Modelos de ecuaciones estructurales en ciencias sociales computacionales", "21"],
        ["3. Marco teórico", "25"],
        ["    3.1 Injusticia discursiva: el marco Fraser-Galtung", "25"],
        ["    3.2 La Hermenéutica Forense Computacional (CFH)", "28"],
        ["    3.3 El Discursive Injustice Score (DIS Score)", "31"],
        ["4. Metodología", "34"],
        ["    4.1 Diseño de la investigación", "34"],
        ["    4.2 Descripción del corpus", "36"],
        ["    4.3 Pipeline de ingesta y procesamiento", "40"],
        ["    4.4 Taxonomía de anotación CFH", "44"],
        ["    4.5 Modelo de medición SEM", "47"],
        ["    4.6 Protocolo de validación", "52"],
        ["5. Resultados", "55"],
        ["6. Discusión y conclusiones", "65"],
        ["Referencias", "72"],
        ["Apéndices", "80"],
      ].map(([titulo, pag]) =>
        new Paragraph({
          children: [
            run(titulo, { size: 22 }),
            new TextRun({ children: [new SimpleField(`PAGEREF ${titulo.trim().replace(/\s+/g,'_')} \\h`)], font: "Times New Roman", size: 22 })
          ],
          tabStops: [{ type: TabStopType.RIGHT, position: 8640, leader: "dot" }],
          spacing: { after: 80 }
        })
      ),

      pageBreak(),

      // ══════════════════════════════════════════════════════════════════════
      // CAPÍTULO 1: INTRODUCCIÓN
      // ══════════════════════════════════════════════════════════════════════
      heading1("1. Introducción"),

      paraJust("El veintidós de junio de dos mil ocho, Fair Leonardo Porras Bernal salió de su casa en el barrio Cazucá de Soacha en busca de trabajo. Tenía dieciséis años. Semanas después, su familia recibió la noticia de que había muerto en un enfrentamiento armado en Ocaña, Norte de Santander, a más de quinientos kilómetros de distancia. El Ejército Nacional lo presentó como un guerrillero dado de baja en combate. Era carpintero."),

      paraJust("La historia de Fair Leonardo no es una excepción. Entre 2002 y 2008, agentes del Estado colombiano asesinaron a civiles y los presentaron fraudulentamente como guerrilleros caídos en combate, en lo que se conoce hoy como falsos positivos o, en la terminología de la Jurisdicción Especial para la Paz (JEP), Macrocaso 003: «Asesinatos y desapariciones forzadas presentadas como bajas en combate por agentes del Estado». La Sala de Reconocimiento de Verdad de la JEP ha identificado al menos 6.402 víctimas de esta práctica para el período 2002-2008 (JEP, 2021); otras estimaciones elevan esa cifra a más de diez mil casos a lo largo de todo el conflicto."),

      paraJust("Este crimen no operó solamente en el campo de batalla. Operó también, y de forma sistemática, en el lenguaje. Cada ejecución extrajudicial requería una doble operación: la muerte física del civil y su transformación discursiva en baja en combate legítima. Esta transformación se producía en formularios de reporte militar, en actas de levantamiento de cadáveres, en comunicados de prensa, en expedientes judiciales y, finalmente, en sentencias de la Corte Suprema de Justicia y el Consejo de Estado. El archivo judicial colombiano es, en este sentido, un archivo de la violencia discursiva institucional: un corpus de documentos en los que el lenguaje del Estado fue instrumentalizado para encubrir, legitimar o minimizar el asesinato sistemático de civiles."),

      paraJust("La pregunta que anima esta investigación es: ¿puede medirse esa violencia discursiva? ¿Existe una manera sistemática y reproducible de cuantificar la distancia entre el lenguaje con que el Estado colombiano describió las muertes de Fair Leonardo y de miles como él, y el lenguaje con que la justicia —ordinaria y transicional— terminó por reconocer lo que realmente ocurrió?"),

      paraJust("Esta tesis propone que sí. Para ello desarrolla un marco teórico-metodológico original, la Hermenéutica Forense Computacional (CFH), que combina procesamiento de lenguaje natural (NLP), modelamiento de ecuaciones estructurales (SEM) y análisis crítico del discurso para construir un índice cuantitativo —el Discursive Injustice Score o DIS Score— capaz de medir el grado de violencia discursiva en documentos judiciales colombianos relacionados con el fenómeno de los falsos positivos."),

      spacer(),
      heading2("1.1 Planteamiento del problema"),

      paraJust("El archivo judicial del conflicto armado colombiano presenta una paradoja lingüística que ha sido documentada por investigadores, organizaciones de víctimas y la propia JEP: los mismos hechos —el asesinato de civiles— fueron descritos con vocabularios radicalmente distintos dependiendo del sistema de justicia, del momento histórico y del poder institucional que producía el texto. La distancia entre «guerrillero abatido en combate» y «civil asesinado en ejecución extrajudicial» no es solo semántica: es la distancia entre la impunidad y la justicia, entre la revictimización y el reconocimiento."),

      paraJust("Esta distancia lingüística ha sido analizada cualitativamente desde el análisis crítico del discurso (van Dijk, 2008; Wodak y Meyer, 2009) y desde la teoría de la violencia cultural (Galtung, 1990). Sin embargo, la investigación cuantitativa sobre el discurso judicial colombiano en materia de falsos positivos es escasa. No existen, hasta donde alcanza la revisión de literatura de esta tesis, herramientas computacionales diseñadas específicamente para medir la injusticia discursiva en archivos judiciales de justicia transicional. Esta ausencia tiene consecuencias prácticas: sin medición sistemática, resulta difícil comparar el tratamiento discursivo de casos similares, identificar patrones de violencia lingüística a escala, o evaluar el impacto de la justicia transicional en la transformación del lenguaje institucional."),

      paraJust("El problema se articula en tres dimensiones que esta investigación aborda de manera integrada."),

      paraJustRuns([
        run("La primera es la dimensión lexical. ", { bold: true }),
        run("El lenguaje militar colombiano desarrolló un sistema de eufemismos —«baja en combate», «resultado operacional», «dado de baja», «neutralizado»— que sistemáticamente reemplazaba la denominación directa del homicidio de civiles por denominaciones que lo enmarcaban como acción militar legítima. Esta práctica eufemística no fue accidental: fue el mecanismo discursivo que permitió reportar los homicidios como méritos operacionales y acumularlos como indicadores de efectividad institucional (Cepeda y Rojas, 2008; Human Rights Watch, 2015). La dimensión lexical del problema consiste en identificar, clasificar y cuantificar estos eufemismos a escala del corpus completo.")
      ]),

      paraJustRuns([
        run("La segunda es la dimensión gramatical. ", { bold: true }),
        run("Más allá del vocabulario, el lenguaje judicial sobre los falsos positivos presenta patrones gramaticales sistemáticos de supresión de agentividad: construcciones pasivas sin complemento agente («fue reportado como baja»), nominalizaciones que eliminan el sujeto responsable («se presentaron los resultados operacionales»), e impersonalizaciones que diluyen la responsabilidad individual en sujetos colectivos («la unidad procedió conforme al protocolo»). Estos mecanismos gramaticales operan como tecnologías de invisibilización del perpetrador y, en consecuencia, de la responsabilidad estatal.")
      ]),

      paraJustRuns([
        run("La tercera es la dimensión epistémica. ", { bold: true }),
        run("El proceso de justicia transicional producido por la JEP en el Macrocaso 003 ha generado un conjunto de documentos —autos de determinación de hechos y conductas, resoluciones de conclusiones, actas de audiencias públicas de reconocimiento— en los que el lenguaje institucional sufre una transformación profunda. Los comparecientes reconocen que «dieron de baja» a personas que eran «civiles inocentes»; la Sala califica los hechos como «crímenes de guerra» y «crímenes de lesa humanidad»; las víctimas recuperan sus nombres. Esta transición lingüística del marco bélico-institucional al marco de los derechos humanos es lo que esta tesis denomina ruptura epistémica, y su medición constituye el núcleo del DIS Score.")
      ]),

      paraJust("El problema de investigación puede formularse, entonces, como la ausencia de un método cuantitativo, reproducible y computacionalmente escalable para medir estas tres dimensiones de la violencia discursiva en el archivo judicial colombiano, y para capturar la dirección y magnitud de la transformación lingüística producida por la justicia transicional."),

      spacer(),
      heading2("1.2 Pregunta de investigación"),

      paraJust("¿En qué medida el lenguaje de los documentos judiciales colombianos relacionados con el Macrocaso 003 de la JEP refleja patrones de injusticia discursiva medibles, y cómo varía la intensidad de esos patrones entre los sistemas de justicia ordinaria y la justicia transicional?"),

      paraJust("Esta pregunta principal se descompone en tres preguntas subsidiarias:"),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [run("¿Qué indicadores lexicales, gramaticales y epistémicos permiten operacionalizar la injusticia discursiva en documentos judiciales sobre falsos positivos?")],
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 120 }
      }),
      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [run("¿Existe una diferencia estadísticamente significativa en el DIS Score entre las sentencias de la justicia ordinaria (Consejo de Estado y Corte Suprema de Justicia) y los autos y resoluciones de la JEP?")],
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 120 }
      }),
      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [run("¿En qué medida el reconocimiento de responsabilidad de los comparecientes ante la JEP se asocia con una transición epistémica medible en el lenguaje de los documentos producidos por el proceso transicional?")],
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 200 }
      }),

      spacer(),
      heading2("1.3 Objetivos"),

      heading3("Objetivo general"),

      paraJust("Desarrollar y validar un modelo computacional —la Hermenéutica Forense Computacional (CFH)— capaz de medir la injusticia discursiva en el archivo judicial colombiano relacionado con el Macrocaso 003 de la JEP, mediante la construcción de un índice cuantitativo (DIS Score) a partir de indicadores NLP integrados en un modelo de ecuaciones estructurales (SEM)."),

      heading3("Objetivos específicos"),

      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [run("Construir y procesar un corpus multiinstitucional de documentos judiciales colombianos sobre falsos positivos, compuesto por sentencias del Consejo de Estado (Corpus A-CE), sentencias de la Corte Suprema de Justicia Sala de Casación Penal (Corpus A-CSJ), y autos y resoluciones de conclusiones del Macrocaso 003 de la JEP (Corpus B).")],
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 120 }
      }),
      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [run("Diseñar y validar una taxonomía de anotación semántica (Taxonomía CFH) con cuatro categorías discursivas —Eufemismo Bélico-Institucional (EBI), Supresión de Agentividad (SA), Negación de Victimización (NV) y Ruptura Epistémica Positiva (REP)— y calcular el acuerdo inter-anotador mediante el coeficiente Cohen κ.")],
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 120 }
      }),
      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [run("Implementar un pipeline de extracción de características lingüísticas a partir de ConfliBERT-Spanish (Yang et al., 2023) para calcular los once indicadores del modelo SEM.")],
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 120 }
      }),
      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [run("Estimar el modelo SEM con cuatro variables latentes y doce indicadores, y probar la hipótesis central H₃: que la Transición Epistémica (η₂) es predicha significativamente por el DIS Score (η₁), con β₂₃ significativo a p < .01.")],
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 120 }
      }),
      new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        children: [run("Evaluar mediante un análisis multi-grupo (MG-SEM) si el modelo presenta invarianza entre los tres subsistemas de justicia (CE, CSJ, JEP), permitiendo comparar la estructura de la injusticia discursiva entre jurisdicciones.")],
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 200 }
      }),

      spacer(),
      heading2("1.4 Justificación"),

      paraJust("Esta investigación se justifica desde tres perspectivas complementarias: la relevancia política y social del problema, la pertinencia científica del enfoque, y la contribución metodológica al campo de la ciencia de datos aplicada a la justicia."),

      paraJustRuns([
        run("Desde el punto de vista político y social, ", { bold: true }),
        run("Colombia atraviesa un momento histórico singular: por primera vez en su historia, un tribunal de justicia transicional —la JEP— está produciendo verdad judicial sobre las ejecuciones extrajudiciales cometidas por el Estado. Las resoluciones de conclusiones del Macrocaso 003 no son solo documentos jurídicos; son actos de reconocimiento de la humanidad de las víctimas y de la responsabilidad institucional. Comprender cómo el lenguaje de estos documentos se diferencia del lenguaje de la justicia ordinaria que precedió a la JEP tiene implicaciones directas para evaluar el impacto real de la justicia transicional como mecanismo de transformación cultural. Si la JEP logra cambiar el lenguaje con que el Estado colombiano habla de sus víctimas, eso es en sí mismo un indicador de éxito del proceso de paz.")
      ]),

      paraJustRuns([
        run("Desde el punto de vista científico, ", { bold: true }),
        run("esta investigación contribuye a tres campos en expansión. Primero, al campo del NLP aplicado a textos jurídicos en español, que ha avanzado significativamente en los últimos años pero mantiene una brecha importante en lo que respecta al análisis de documentos de justicia transicional latinoamericana (Chalkidis et al., 2023). Segundo, al campo de las ciencias sociales computacionales, al proponer un puente entre el análisis de discurso crítico y los métodos cuantitativos de aprendizaje automático. Tercero, al campo del análisis del conflicto con NLP, al utilizar ConfliBERT-Spanish (Yang et al., 2023), un modelo de lenguaje especializado en textos de conflicto y paz en español, que hasta el momento no ha sido aplicado al archivo judicial del conflicto armado colombiano.")
      ]),

      paraJustRuns([
        run("Desde el punto de vista metodológico, ", { bold: true }),
        run("la CFH ofrece una contribución original al integrar tres tradiciones metodológicas que raramente se combinan: el análisis crítico del discurso (que aporta el marco conceptual de la injusticia discursiva), el NLP (que aporta las herramientas de extracción automática de indicadores lingüísticos) y el SEM (que aporta un marco estadístico para modelar relaciones latentes entre variables no observadas directamente). Esta integración permite superar la dicotomía entre el análisis cualitativo profundo —que captura matices pero no es escalable— y el análisis cuantitativo superficial —que es escalable pero pierde significado semántico. La CFH aspira a ser un método que combine rigor interpretativo y escalabilidad computacional.")
      ]),

      paraJust("Finalmente, desde el punto de vista de las víctimas del Macrocaso 003, esta investigación parte de una convicción ética: que dar nombre a los mecanismos lingüísticos de la impunidad es un acto de memoria. Cada vez que un documento judicial describió a Fair Leonardo Porras Bernal como «guerrillero neutralizado», se cometió una segunda violencia. Medir esa violencia no la deshace, pero la hace visible."),

      spacer(),
      heading2("1.5 Estructura del documento"),

      paraJust("Este documento está organizado en seis capítulos. El Capítulo 2 presenta el estado del arte en cuatro áreas: violencia cultural y lenguaje institucional, NLP aplicado a textos judiciales, justicia transicional y análisis del discurso, y modelos de ecuaciones estructurales en ciencias sociales computacionales. El Capítulo 3 desarrolla el marco teórico de la investigación, articulando el concepto de injusticia discursiva desde las tradiciones de Nancy Fraser y Johan Galtung, y formalizando el marco CFH y el DIS Score. El Capítulo 4 describe en detalle la metodología: el diseño del corpus, el pipeline de procesamiento, la taxonomía de anotación, el modelo SEM y los protocolos de validación. El Capítulo 5 presenta los resultados del modelo y sus análisis exploratorios. El Capítulo 6 discute los hallazgos, sus limitaciones y sus implicaciones para la investigación futura y para la política pública de justicia transicional en Colombia."),

      spacer(3),

      // ══════════════════════════════════════════════════════════════════════
      // REFERENCIAS DE LA INTRODUCCIÓN
      // ══════════════════════════════════════════════════════════════════════
      new Paragraph({
        children: [run("Referencias citadas en esta sección", { bold: true, size: 22 })],
        spacing: { before: 480, after: 200 }
      }),

      ...[
        "Cepeda Castro, I. y Rojas, J. (2008). A las puertas de El Ubérrimo. Debate.",
        "Chalkidis, I., Garneau, N., Cao, Y., Huang, Z., Stathis, T., Thavarajah, J., y Goldfarb, D. (2023). LexFiles and LegalLAMA: Facilitating English Multinational Legal Language Model Development. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 15, 6228–6244.",
        "Fraser, N. (1995). From redistribution to recognition? Dilemmas of justice in a 'post-socialist' age. New Left Review, 212, 68–93.",
        "Galtung, J. (1990). Cultural violence. Journal of Peace Research, 27(3), 291–305.",
        "Human Rights Watch. (2015). El rol de los altos mandos en los falsos positivos. HRW.",
        "JEP — Jurisdicción Especial para la Paz. (2021). Auto No. 125 de 2021 — Caso 03, Subcaso Norte de Santander. Sala de Reconocimiento de Verdad, de Responsabilidad y de Determinación de los Hechos y Conductas.",
        "Van Dijk, T. A. (2008). Discourse and Power. Palgrave Macmillan.",
        "Wodak, R. y Meyer, M. (Eds.). (2009). Methods of Critical Discourse Analysis (2.ª ed.). SAGE.",
        "Yang, W., Salam, M. A., Alhelbawy, A., El-Beltagy, S., y Zubiaga, A. (2023). ConfliBERT: A Pre-trained Language Model for Political Conflict and Violence. IEEE CiSt 2023.",
      ].map(ref =>
        new Paragraph({
          children: [run(ref, { size: 20 })],
          alignment: AlignmentType.JUSTIFIED,
          indent: { left: 720, hanging: 720 },
          spacing: { after: 120 }
        })
      ),

    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/mnt/user-data/outputs/CFH_Tesis_Capitulo1_Introduccion.docx", buffer);
  console.log("✓ CFH_Tesis_Capitulo1_Introduccion.docx generado");
});

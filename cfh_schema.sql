-- ============================================================================
-- cfh_schema.sql  (v2 — granularidad mixta)
-- ============================================================================
-- Cambios respecto a v1:
--   1. `bloques` ahora soporta tres granularidades:
--      'seccion'           — sección de documento (HECHOS, RECONOCIMIENTO...)
--      'bloque_granular'   — un .txt individual de Corpus B (2.641 archivos)
--      'bloque_audiencia'  — bloque temporal del Corpus C
--   2. `bloques.bloque_padre_id` permite vincular un bloque granular a su
--      sección padre.
--   3. Nuevos campos `identificador_externo` y `ruta_archivo` para trazar al
--      CSV o al .txt original.
--
-- Uso (recreación limpia desde Anaconda Prompt):
--   python -c "import sqlite3, os; os.path.exists('cfh.db') and os.remove('cfh.db'); con = sqlite3.connect('cfh.db'); con.executescript(open('cfh_schema.sql', encoding='utf-8').read()); print('OK')"
-- ============================================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA encoding = "UTF-8";

-- ============================================================================
-- 1. CORPORA
-- ============================================================================
CREATE TABLE corpora (
    codigo          TEXT PRIMARY KEY,
    nombre          TEXT NOT NULL,
    descripcion     TEXT,
    periodo_inicio  INTEGER,
    periodo_fin     INTEGER,
    es_referencia   INTEGER NOT NULL DEFAULT 0,
    creado_en       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO corpora (codigo, nombre, descripcion, periodo_inicio, periodo_fin, es_referencia) VALUES
    ('A-CE',        'Consejo de Estado',           'Reparación directa, justicia ordinaria',                  1994, 2021, 0),
    ('A-CSJ',       'Corte Suprema, Sala Penal',   'Casación penal, justicia ordinaria',                      2012, 2020, 0),
    ('B-JEP',       'JEP escrito',                 'Autos y resoluciones SRVR — Macrocaso 003',               2021, 2025, 0),
    ('C-JEP-oral',  'JEP audiencias orales',       'Audiencias de reconocimiento — Macrocaso 003',            2022, 2024, 0),
    ('REF-MAFAPO',  'MAFAPO',                      'Centroide de referencia — testimonios públicos',          NULL, NULL, 1),
    ('REF-CIDH',    'CIDH',                        'Centroide de referencia — sentencia Villamizar Durán',    NULL, NULL, 1);


-- ============================================================================
-- 2. DOCUMENTOS
-- ============================================================================
CREATE TABLE documentos (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus              TEXT NOT NULL,
    doc_id_externo      TEXT,
    titulo              TEXT NOT NULL,
    tipo_documento      TEXT,
    radicado            TEXT,
    fecha               TEXT,
    año                 INTEGER,
    fuente_org          TEXT,
    magistrado_ponente  TEXT,
    departamento        TEXT,
    municipio           TEXT,
    batallon            TEXT,
    n_paginas           INTEGER,
    n_chars             INTEGER,
    ruta_original       TEXT UNIQUE,
    sha256              TEXT,
    creado_en           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (corpus) REFERENCES corpora(codigo)
);

CREATE INDEX idx_documentos_corpus ON documentos(corpus);
CREATE INDEX idx_documentos_año ON documentos(año);
CREATE INDEX idx_documentos_doc_id_externo ON documentos(doc_id_externo);


-- ============================================================================
-- 3. BLOQUES — granularidad mixta
-- ============================================================================
CREATE TABLE bloques (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    documento_id            INTEGER NOT NULL,
    granularidad            TEXT NOT NULL,
    seccion                 TEXT,
    orden                   INTEGER NOT NULL,
    texto                   TEXT NOT NULL,
    n_tokens                INTEGER,
    n_chars                 INTEGER,
    bloque_padre_id         INTEGER,
    identificador_externo   TEXT,
    ruta_archivo            TEXT,
    embedding_path          TEXT,
    creado_en               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (documento_id) REFERENCES documentos(id) ON DELETE CASCADE,
    FOREIGN KEY (bloque_padre_id) REFERENCES bloques(id)
);

CREATE INDEX idx_bloques_documento ON bloques(documento_id);
CREATE INDEX idx_bloques_seccion ON bloques(seccion);
CREATE INDEX idx_bloques_granularidad ON bloques(granularidad);
CREATE INDEX idx_bloques_padre ON bloques(bloque_padre_id);
CREATE INDEX idx_bloques_id_externo ON bloques(identificador_externo);


-- ============================================================================
-- 4. AUDIENCIAS
-- ============================================================================
CREATE TABLE audiencias (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    documento_id    INTEGER NOT NULL UNIQUE,
    subcaso         TEXT NOT NULL UNIQUE,
    fecha_inicio    TEXT,
    duracion_horas  REAL,
    magistrada      TEXT,
    materialidad    TEXT,
    ruta_audio      TEXT,
    ruta_video      TEXT,
    drm_bloqueado   INTEGER NOT NULL DEFAULT 0,
    notas           TEXT,
    FOREIGN KEY (documento_id) REFERENCES documentos(id) ON DELETE CASCADE
);


-- ============================================================================
-- 5. COMPARECIENTES
-- ============================================================================
CREATE TABLE comparecientes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    audiencia_id    INTEGER NOT NULL,
    nombre          TEXT,
    rango           TEXT,
    rol_jep         TEXT,
    speaker_id      TEXT,
    UNIQUE (audiencia_id, speaker_id),
    FOREIGN KEY (audiencia_id) REFERENCES audiencias(id) ON DELETE CASCADE
);

CREATE INDEX idx_comparecientes_audiencia ON comparecientes(audiencia_id);


-- ============================================================================
-- 6. RUNS — antes que segmentos_orales para FK
-- ============================================================================
CREATE TABLE runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    script          TEXT NOT NULL,
    git_commit      TEXT,
    parametros_json TEXT,
    descripcion     TEXT,
    estado          TEXT NOT NULL DEFAULT 'completado'
);


-- ============================================================================
-- 7. MODELOS
-- ============================================================================
CREATE TABLE modelos (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre          TEXT NOT NULL,
    version         TEXT NOT NULL,
    tipo            TEXT NOT NULL,
    ruta_artifact   TEXT,
    metricas_json   TEXT,
    referencia_bib  TEXT,
    notas           TEXT,
    creado_en       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (nombre, version)
);

INSERT INTO modelos (nombre, version, tipo, referencia_bib, metricas_json) VALUES
    ('CFH-BERT',                   'v1',          'classifier',         'Camacho 2026 (este trabajo)',                  '{"f1_macro": 0.27}'),
    ('CFH-BERT',                   'v2',          'classifier',         'Camacho 2026 (este trabajo)',                  '{"f1_macro": 0.58, "f1_REP": 0.77, "f1_NV": 0.32, "n_anotaciones": 100}'),
    ('ConfliBERT-Spanish',         'beto-cased-v1','embedder',           'Yang et al. 2023, IEEE CiSt',                  NULL),
    ('BETO',                       'cased',       'embedder',           'Cañete et al. 2020',                           NULL),
    ('MediaPipe-FaceLandmarker',   'tasks-2023',  'extractor_facial',   'Lugaresi et al. 2023, CVPR Workshop',          NULL),
    ('OpenSMILE-eGeMAPS',          'v02',         'extractor_acustico', 'Eyben et al. 2016',                            '{"n_features": 88}'),
    ('Whisper',                    'large-v3',    'asr',                'Radford et al. 2023, OpenAI',                  NULL),
    ('pyannote-audio',             '2.1',         'diarizer',           'Bredin et al. 2023, ICASSP',                   NULL),
    ('Pipeline-Capa1',             'cap5-v15',    'classifier',         'Camacho 2026 — pipeline interno cap.5',        NULL),
    ('Pipeline-Beach',             'y11-y12-y13', 'classifier',         'Beach et al. 2021 adaptado al español jurídico', NULL);


-- ============================================================================
-- 8. SEGMENTOS_ORALES
-- ============================================================================
CREATE TABLE segmentos_orales (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    audiencia_id        INTEGER NOT NULL,
    compareciente_id    INTEGER,
    bloque_id           INTEGER,
    t_inicio            REAL NOT NULL,
    t_fin               REAL NOT NULL,
    duracion            REAL,
    speaker_diarizacion TEXT,
    -- Action Units faciales
    au1                 REAL,
    au4                 REAL,
    au6                 REAL,
    au12                REAL,
    au15                REAL,
    au17                REAL,
    -- Prosodia
    f0_mean             REAL,
    f0_stddev           REAL,
    loudness_mean       REAL,
    jitter              REAL,
    shimmer             REAL,
    hnr                 REAL,
    egemaps_full_json   TEXT,
    -- ICM
    icm_facial          REAL,
    icm_vocal           REAL,
    icm_verbal          REAL,
    icm_tri             REAL,
    -- Trazabilidad
    fuente_csv          TEXT,
    run_id              INTEGER,
    FOREIGN KEY (audiencia_id) REFERENCES audiencias(id) ON DELETE CASCADE,
    FOREIGN KEY (compareciente_id) REFERENCES comparecientes(id),
    FOREIGN KEY (bloque_id) REFERENCES bloques(id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX idx_segmentos_audiencia ON segmentos_orales(audiencia_id);
CREATE INDEX idx_segmentos_compareciente ON segmentos_orales(compareciente_id);


-- ============================================================================
-- 9. INDICADORES — tabla larga
-- ============================================================================
CREATE TABLE indicadores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    bloque_id       INTEGER NOT NULL,
    codigo          TEXT NOT NULL,
    valor           REAL,
    valor_norm      REAL,
    modelo_id       INTEGER,
    run_id          INTEGER,
    notas           TEXT,
    creado_en       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (bloque_id) REFERENCES bloques(id) ON DELETE CASCADE,
    FOREIGN KEY (modelo_id) REFERENCES modelos(id),
    FOREIGN KEY (run_id) REFERENCES runs(id),
    UNIQUE (bloque_id, codigo, modelo_id, run_id)
);

CREATE INDEX idx_indicadores_bloque ON indicadores(bloque_id);
CREATE INDEX idx_indicadores_codigo ON indicadores(codigo);
CREATE INDEX idx_indicadores_run ON indicadores(run_id);


-- ============================================================================
-- 10. ANOTACIONES
-- ============================================================================
CREATE TABLE anotaciones (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    bloque_id               INTEGER,
    label                   TEXT NOT NULL,
    span_inicio             INTEGER,
    span_fin                INTEGER,
    span_texto              TEXT,
    anotador                TEXT NOT NULL,
    label_studio_id         INTEGER,
    inner_id                INTEGER,
    n_spans                 INTEGER,
    etiquetas_combinadas    TEXT,
    es_resumen              INTEGER NOT NULL DEFAULT 0,
    creado_en               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (bloque_id) REFERENCES bloques(id) ON DELETE CASCADE
);

CREATE INDEX idx_anotaciones_bloque ON anotaciones(bloque_id);
CREATE INDEX idx_anotaciones_label ON anotaciones(label);
CREATE INDEX idx_anotaciones_anotador ON anotaciones(anotador);


-- ============================================================================
-- 11. CENTROIDES
-- ============================================================================
CREATE TABLE centroides (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre          TEXT NOT NULL UNIQUE,
    corpus_origen   TEXT NOT NULL,
    n_textos        INTEGER NOT NULL,
    modelo_id       INTEGER NOT NULL,
    ruta_npy        TEXT,
    descripcion     TEXT,
    creado_en       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (corpus_origen) REFERENCES corpora(codigo),
    FOREIGN KEY (modelo_id) REFERENCES modelos(id)
);


-- ============================================================================
-- VISTAS
-- ============================================================================

CREATE VIEW v_tabla_5_1 AS
SELECT
    c.codigo                          AS corpus,
    COUNT(DISTINCT d.id)              AS n_documentos,
    COUNT(DISTINCT b.id)              AS n_bloques_total,
    COUNT(DISTINCT CASE WHEN b.granularidad = 'seccion' THEN b.id END)          AS n_secciones,
    COUNT(DISTINCT CASE WHEN b.granularidad = 'bloque_granular' THEN b.id END)  AS n_bloques_granulares,
    COUNT(DISTINCT CASE WHEN b.granularidad = 'bloque_audiencia' THEN b.id END) AS n_bloques_audiencia,
    MIN(d.año)                        AS año_inicio,
    MAX(d.año)                        AS año_fin
FROM corpora c
LEFT JOIN documentos d ON d.corpus = c.codigo
LEFT JOIN bloques b    ON b.documento_id = d.id
WHERE c.es_referencia = 0
GROUP BY c.codigo
ORDER BY c.codigo;


CREATE VIEW v_indicadores_por_corpus AS
SELECT
    d.corpus,
    i.codigo,
    m.nombre || ' ' || m.version  AS modelo,
    r.id                          AS run_id,
    COUNT(i.id)                   AS n,
    AVG(i.valor)                  AS media,
    MIN(i.valor)                  AS minimo,
    MAX(i.valor)                  AS maximo
FROM indicadores i
JOIN bloques b      ON b.id = i.bloque_id
JOIN documentos d   ON d.id = b.documento_id
LEFT JOIN modelos m ON m.id = i.modelo_id
LEFT JOIN runs r    ON r.id = i.run_id
GROUP BY d.corpus, i.codigo, m.id, r.id
ORDER BY d.corpus, i.codigo, r.id;


CREATE VIEW v_audiencias_resumen AS
SELECT
    a.subcaso,
    a.fecha_inicio,
    a.duracion_horas,
    a.drm_bloqueado,
    COUNT(DISTINCT c.id)               AS n_comparecientes,
    COUNT(DISTINCT s.id)               AS n_segmentos_orales,
    COUNT(DISTINCT b.id)               AS n_bloques
FROM audiencias a
LEFT JOIN comparecientes c   ON c.audiencia_id = a.id
LEFT JOIN segmentos_orales s ON s.audiencia_id = a.id
LEFT JOIN documentos d       ON d.id = a.documento_id
LEFT JOIN bloques b          ON b.documento_id = d.id
GROUP BY a.id;
